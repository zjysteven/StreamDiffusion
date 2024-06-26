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
parser.add_argument('--cond_size', type=int, default=64)
parser.add_argument('--frame_bff_size', type=int, default=1)
parser.add_argument('--num_images', type=int, default=50000)
parser.add_argument(
    '--lcm_id', type=str, default='1e-6',
    choices=[
        '1e-6', '5e-6', '1e-5', '5e-5', '1e-4'
        '1e-6-cfg7.5', '5e-6-cfg7.5', '1e-5-cfg7.5', '5e-5-cfg7.5', '1e-4-cfg7.5'      
    ]                    
)
parser.add_argument('--server', type=str, choices=['athena', 'lighthouse'], default='lighthouse')
args = parser.parse_args()

#################################################################
# model loading
controlnet = ControlNetModel.from_pretrained(
    'lllyasviel/control_v11f1e_sd15_tile' if args.size == 512 else 'zjysteven/control_minisd_tile',
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5" if args.size == 512 else "lambdalabs/miniSD-diffusers",
    controlnet=controlnet,
    torch_dtype=torch.float16,
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
    frame_buffer_size=args.frame_bff_size,
)
delay = stream.denoising_steps_num

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora(
    "latent-consistency/lcm-lora-sdv1-5" if args.size == 512 else f"zjysteven/lcm-lora-miniSD-{args.lcm_id}"
)
stream.fuse_lora()
# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

if args.accel == 'trt':
    stream = accelerate_with_tensorrt_unetcontrol(
        stream,
        f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}" \
        if args.size != 256 else \
        f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}_lcm{args.lcm_id}", 
        max_batch_size=stream.batch_size,
        engine_build_options={
            'opt_image_height': args.size,
            'opt_image_width': args.size, 
        }
        # f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}", 
        # max_batch_size=stream.denoising_steps_num * 16, # max frame bff size is 128 on A100
    )

# data
if args.server == 'lighthouse':
    anno_path = '/home/public/coco-stuff/annotations/captions_val2017.json'
elif args.server == 'athena':
    anno_path = '/data/coco/annotations/captions_val2017.json'
with open(anno_path, 'r') as f:
    annos = json.load(f)

id2img = {}
img2id = {}
for tmp in annos['images']:
    id2img[tmp['id']] = tmp['file_name']
    img2id[tmp['file_name']] = tmp['id']

img2caps = defaultdict(list)
for tmp in annos['annotations']:
    img2caps[id2img[tmp['image_id']]].append(tmp['caption'])

if args.server == 'lighthouse':
    data_path = '/home/public/coco-stuff/images/val2017'
elif args.server == 'athena':
    data_path = '/data/coco/val2017'
all_images = os.listdir(data_path)
np.random.seed(0)
selected = np.random.choice(all_images, min(len(all_images), args.num_images), replace=False).tolist()
selected = selected + [selected[-1]] * (delay - 1)

save_dir = f'./outputs/controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_accel={args.accel}_framebff{args.frame_bff_size}'
os.makedirs(save_dir, exist_ok=True)

input_buffer = []
cond_buffer = []
img_batch_size = stream.frame_bff_size
num_batches = len(selected) // img_batch_size
img_cnt = 0
negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality, blur, blurred, DOF"] * img_batch_size

for i in tqdm(range(num_batches), total=num_batches):
    filenames = selected[i*img_batch_size:(i+1)*img_batch_size]

    prompts = []
    for filename in filenames:
        prompts.append(f"{img2caps[filename][0]}, realistic, best quality, extremely detailed")
        original = load_image(
            os.path.join(data_path, filename)
        ).resize((args.size, args.size))
        cond = downsample(original, (args.cond_size, args.cond_size))
        cond_resized = cond.resize((args.size, args.size))
        input_buffer.append(original)
        cond_buffer.append(cond_resized)

    if i == 0:
        stream.prepare(
            prompts,
            negative_prompt,
            guidance_scale=1.2,
        )
    else:
        stream.update_prompt(prompts)
    
    images = stream(cond_buffer[-img_batch_size:])

    if i >= delay - 1:
        for j in range(len(images)):
            grid_output = make_image_grid(
                [
                    input_buffer.pop(0),
                    cond_buffer.pop(0),
                    postprocess_image(images[j:j+1])[0],
                ], 
                rows=1, cols=3
            )
            grid_output.save(f'{save_dir}/img_{img_cnt:04d}.jpg')
            img_cnt += 1

with open(f'{save_dir}/ema_time.txt', 'w') as f:
    f.write(f'Per batch: {stream.inference_time_ema:.6f} s\n')
    f.write(f'Per image: {stream.inference_time_ema/img_batch_size:.6f} s\n')
    f.write(f'FPS: {1/(stream.inference_time_ema/img_batch_size):.4f}')