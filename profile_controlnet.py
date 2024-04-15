import argparse
import os
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch
from diffusers import (
    AutoencoderTiny, LCMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
)
from diffusers.utils import load_image, make_image_grid
from img_utils import downsample, tensor2img
from PIL import Image

from streamdiffusion import StreamUNetControlDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt_unetcontrol

parser = argparse.ArgumentParser()
parser.add_argument('--accel', type=str, choices=['xformers', 'trt'], default='xformers')
parser.add_argument('--num_inference_steps', type=int, default=4)
parser.add_argument('--strength', type=float, default=0.8)
parser.add_argument('--cfg_type', type=str, choices=['none', 'self', 'initialize', 'full'], default='none')
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--scale', type=int, choices=[2, 4, 8], default=8)
args = parser.parse_args()

#################################################################
# model loading
controlnet = ControlNetModel.from_pretrained(
    'lllyasviel/control_v11f1e_sd15_tile', 
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Enable acceleration
if args.accel == 'xformers':
    pipe.enable_xformers_memory_efficient_attention()
#################################################################

stream = StreamUNetControlDiffusion(
    pipe,
    height=args.size,
    width=args.size,
    num_inference_steps=args.num_inference_steps,
    strength=args.strength,
    torch_dtype=torch.float16,
    cfg_type=args.cfg_type,
)
delay = stream.batch_size

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora("latent-consistency/lcm-lora-sdv1-5")
stream.fuse_lora()
# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

if args.accel == 'trt':
    stream = accelerate_with_tensorrt_unetcontrol(
        stream, 
        f"engines/unet_controlnet_size{args.size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}", 
        max_batch_size=stream.batch_size,
    )

# data
with open('/home/public/coco-stuff/annotations/captions_val2017.json', 'r') as f:
    annos = json.load(f)

id2img = {}
img2id = {}
for tmp in annos['images']:
    id2img[tmp['id']] = tmp['file_name']
    img2id[tmp['file_name']] = tmp['id']

img2caps = defaultdict(list)
for tmp in annos['annotations']:
    img2caps[id2img[tmp['image_id']]].append(tmp['caption'])

num_images = 1000
all_images = os.listdir('/home/public/coco-stuff/images/val2017')
np.random.seed(0)
selected = np.random.choice(all_images, min(len(all_images), num_images), replace=False).tolist()
selected = selected + [selected[-1]] * (delay - 1)

save_dir = f'./outputs/controlnet_size{args.size}_scale{args.scale}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_accel={args.accel}'
os.makedirs(save_dir, exist_ok=True)

negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

input_buffer = []
cond_buffer = []
for i, filename in tqdm(enumerate(selected), total=len(selected)):
    prompt = f"{img2caps[filename][0]}, realistic, best quality, extremely detailed"
    if i == 0:
        stream.prepare(
            prompt,
            negative_prompt,
            guidance_scale=1.2,
        )
    else:
        stream.update_prompt(prompt)
    
    original = load_image(
        os.path.join('/home/public/coco-stuff/images/val2017', filename)
    ).resize((args.size, args.size))
    cond = downsample(original, args.scale)
    cond_resized = cond.resize((args.size, args.size))
    input_buffer.append(original)
    cond_buffer.append(cond_resized)

    image, _ = stream(cond)

    if i >= delay - 1:
        grid_output = make_image_grid(
            [
                input_buffer.pop(0),
                cond_buffer.pop(0),
                postprocess_image(image)[0],
            ], 
            rows=1, cols=3
        )
        grid_output.save(f'{save_dir}/img_{i-delay+1:04d}.jpg')

with open(f'{save_dir}/ema_time.txt', 'w') as f:
    f.write(f'{stream.inference_time_ema:.6f} s')