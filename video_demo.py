import argparse
import os
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch
from torchvision.io import read_video, write_video
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
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--cond_size', type=int, default=64)
parser.add_argument('--frame_bff_size', type=int, default=1)
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument(
    '--lcm_id', type=str, default='1e-6',
    choices=[
        '1e-6', '5e-6', '1e-5', '5e-5', '1e-4'
        '1e-6-cfg7.5', '5e-6-cfg7.5', '1e-5-cfg7.5', '5e-5-cfg7.5', '1e-4-cfg7.5'      
    ]                    
)
parser.add_argument('--server', type=str, choices=['athena', 'lighthouse'], default='lighthouse')
args = parser.parse_args()

# input data
video_info = read_video(args.video_path)
video = video_info[0] / 255
fps = int(video_info[2]["video_fps"])
print(f"original video shape: {video.shape}")
num_frames = (video.shape[0] // fps) * fps
args.frame_bff_size = fps # hard-coded, is there a better way?
assert num_frames % args.frame_bff_size == 0, \
    f"num_frames [{num_frames}] should be divisible by frame_bff_size [{args.frame_bff_size}]"

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
similar_image_filter_threshold = 0.90
similar_image_filter_max_skip_frame = 30
stream.enable_similar_image_filter(
    similar_image_filter_threshold,
    similar_image_filter_max_skip_frame,
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

save_dir = f'./video_outputs/'
os.makedirs(save_dir, exist_ok=True)

input_buffer = []
cond_buffer = []
img_batch_size = stream.frame_bff_size
negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality, blur, blurred, DOF"] * img_batch_size

stream.prepare(
    [args.prompt] * img_batch_size,
    negative_prompt,
    guidance_scale=1.2,
)
video_result = torch.zeros(num_frames, args.size, args.size, 3)
video = list(video) + [video[-1]] * (delay - 1) * img_batch_size
num_batches = len(video) // img_batch_size
frame_cnt = 0

for i in tqdm(range(num_batches), total=num_batches):
    frames = video[i*img_batch_size:(i+1)*img_batch_size]
    cond_buffer.extend(frames)
    
    output_frames, _ = stream(
        torch.stack(cond_buffer[-img_batch_size:]).permute(0, 3, 1, 2)
    )

    if i >= delay - 1:
        for j in range(len(output_frames)):
            video_result[frame_cnt] = postprocess_image(
                output_frames[j].permute(1, 2, 0),
                output_type="pt"
            )
            frame_cnt += 1

#assert frame_cnt == num_frames, f"frame_cnt [{frame_cnt}] != num_frames [{num_frames}]"
print(f"frame_cnt: {frame_cnt}, num_frames: {num_frames}")
video_result = video_result * 255
write_video(
    os.path.join(save_dir, f"{args.video_path.split('/')[-1]}"), 
    video_result, fps=fps
)

with open(f"{save_dir}/{args.video_path.split('/')[-1].split('.')[0]}_ema_time.txt", 'w') as f:
    f.write(f'Per batch: {stream.inference_time_ema:.6f} s\n')
    f.write(f'Per image: {stream.inference_time_ema/img_batch_size:.6f} s\n')
    f.write(f'FPS: {1/(stream.inference_time_ema/img_batch_size):.4f}')