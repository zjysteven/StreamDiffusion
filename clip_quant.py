"""
Quantization of CLIP tokenizer (CLIPTokenizer) and CLIP text encoder (CLIPTextModel).

Based on the NVIDIA Stable Diffusion Quantization guide:
https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/diffusers/README.md

---- Copy of NVIDIA license for software use and modification:

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""

import argparse
import os
import json
import re
from pathlib import Path

from collections import defaultdict
import numpy as np
from tqdm import tqdm

import onnx
import torch
from torch.onnx import export as onnx_export

from diffusers import (
    AutoencoderTiny, LCMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
)

from diffusers.utils import load_image, make_image_grid
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from img_utils import downsample, tensor2img
from PIL import Image

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.calib.max import MaxCalibrator
from optimum.onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data

USE_PEFT = True
try:
    from peft.tuners.lora.layer import Conv2d as PEFTLoRAConv2d
    from peft.tuners.lora.layer import Linear as PEFTLoRALinear
except ModuleNotFoundError:
    USE_PEFT = False


# ========================================================================
# ========================================================================
# =========================== Arg Parse ==================================
# ========================================================================
# ========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--num_inference_steps', type=int, default=4)
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--cond_size', type=int, default=64)
parser.add_argument('--num_images', type=int, default=50000)

parser.add_argument('--server', type=str, choices=['athena', 'lighthouse'], default='lighthouse')
parser.add_argument('--separate_controlnet', action='store_true')

parser.add_argument('--calib_size', type=int, default=512)
parser.add_argument('--quant_level', type=float, default=3.0, choices=[1.0, 2.0, 2.5, 3.0])
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--percentile', type=float, default=1.0)
args = parser.parse_args()


# ========================================================================
# ========================================================================
# =========================== Functions ==================================
# ========================================================================
# ========================================================================
def do_calibration(pipe, calib_list, neg_prompt, num_steps, generator, calib_size):
    for i_th, (prompt, image, control_image) in enumerate(calib_list):
        if i_th >= calib_size:
            return
        
        pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image,
            control_image=control_image,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=1.2,
        ).images


def gen_calib_list(pipe):
    # Load data
    if args.server == 'lighthouse':
        anno_path = '/home/public/coco-stuff/annotations/captions_val2017.json'
        data_path = '/home/public/coco-stuff/images/val2017'
    elif args.server == 'athena':
        anno_path = '/data/coco/annotations/captions_val2017.json'
        data_path = '/data/coco/val2017'
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

    all_images = os.listdir(data_path)
    np.random.seed(0)
    selected = np.random.choice(all_images, min(len(all_images), args.calib_size), replace=False).tolist()

    # Prompts: each element should be a one-element list, e.g. [['abc'], ['def']]
    # Images: Tensor of downsampled images
    # Control Images: Generated from each image using pipe.prepare_control_image
    print('>> Generating calibration items list...')
    calib_list = []
    for i in tqdm(range(len(selected)), total=len(selected)):
        filename = selected[i]
        prompts = [f'{img2caps[filename][0]}, realistic, best quality, extremely detailed']
        original = load_image(os.path.join(data_path, filename)).resize((args.size, args.size))
        cond = downsample(original, (args.cond_size, args.cond_size))
        cond_resized = cond.resize((args.size, args.size))
        
        ctrl = pipe.prepare_control_image(
            image=cond_resized,
            width=args.size,
            height=args.size,
            batch_size=1,
            num_images_per_prompt=1,
            device='cuda',
            dtype=torch.float16,
            do_classifier_free_guidance=True,
            guess_mode=False,
        )[0][None, :]

        calib_list.append((prompts, cond_resized, ctrl))
    print('>> Calibration list generation complete.')

    return calib_list


def check_lora(unet):
    for name, module in unet.named_modules():
        if isinstance(module, (LoRACompatibleConv, LoRACompatibleLinear)):
            assert (
                module.lora_layer is None
            ), f'To quantize {name}, LoRA layer should be fused/merged. Please fuse the LoRA layer before quantization.'
        elif USE_PEFT and isinstance(module, (PEFTLoRAConv2d, PEFTLoRALinear)):
            assert (
                module.merged
            ), f'To quantize {name}, LoRA layer should be fused/merged. Please fuse the LoRa layer before quantization.'


# ========================================================================
# ========================================================================
# =========================== Main Func ==================================
# ========================================================================
# ========================================================================
if __name__ == "__main__":
    # Some constants..
    ONNX_DIR = f'./engines/clip512/onnx'
    QUANT_DIR = f'./clip_quant'

    TOK_SDICT_PATH = os.path.join(QUANT_DIR, 'clip_tokenizer_state_dict_qint8.pt')
    TOK_QUANT_PATH = os.path.join(QUANT_DIR, 'clip_tokenizer_quant_states_qint8.pt')
    ENC_SDICT_PATH = os.path.join(QUANT_DIR, 'clip_text_encoder_state_dict_qint8.pt')
    ENC_QUANT_PATH = os.path.join(QUANT_DIR, 'clip_text_encoder_quant_states_qint8.pt')

    TOK_CKPT_PATH = os.path.join(QUANT_DIR, 'clip_tokenizer_ckpt')
    ENC_CKPT_PATH = os.path.join(QUANT_DIR, 'clip_text_encoder_ckpt')

    MODEL_NAME = 'pt-sk/stable-diffusion-1.5'
    #MODEL_NAME = 'runwayml/stable-diffusion-v1-5'

    if not os.path.exists(QUANT_DIR):
        os.makedirs(QUANT_DIR)

    # Generator for image generation - matches StreamDiffusion pipeline
    generator = torch.Generator()
    generator.manual_seed(2)

    # Load pretrained models
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11f1e_sd15_tile',
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        MODEL_NAME,
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Fuse LoRA layers
    pipe.load_lora_weights('latent-consistency/lcm-lora-sdv1-5')
    pipe.fuse_lora(fuse_unet=True, fuse_text_encoder=True, lora_scale=1.0, safe_fusing=False)

    # Same prompts + negative prompts as our profiling pipeline
    calibration_items = gen_calib_list(pipe)
    neg_prompt = ['monochrome, lowres, bad anatomy, worst quality, low quality, blur, blurred, DOF']
    extra_step = 1

    # Get quant configuration
    quant_config = mtq.INT8_SMOOTHQUANT_CFG
    """
    quant_config = get_int8_config(
        pipe.unet,
        args.quant_level,
        args.alpha,
        args.percentile,
        args.num_inference_steps + extra_step,
        collect_method='global_min',
        num_layers = nlayers,
    )
    """

    # Define forward loops for calibration
    def forward_tokenizer(tok):
        pipe.tokenizer = tok
        do_calibration(
            pipe=pipe,
            calib_list=calibration_items,
            neg_prompt=neg_prompt,
            num_steps=args.num_inference_steps,
            generator=generator,
            calib_size=args.calib_size
        )

    def forward_encoder(enc):
        pipe.text_encoder = enc
        do_calibration(
            pipe=pipe,
            calib_list=calibration_items,
            neg_prompt=neg_prompt,
            num_steps=args.num_inference_steps,
            generator=generator,
            calib_size=args.calib_size
        )

    # All the LoRA layers should be fused - check here
    check_lora(pipe.unet)

    # Calibration + quantization (Tokenizer)
    mtq.quantize(pipe.tokenizer, quant_config, forward_tokenizer)

    print('>> DEBUG << \n\nPrinting tokenizer quant summary:')
    mtq.print_quant_summary(pipe.tokenizer)
    print('\n\n')

    torch.save(mto.modelopt_state(pipe.tokenizer), TOK_QUANT_PATH)
    torch.save(pipe.tokenizer.state_dict(), TOK_SDICT_PATH)
    mto.save(pipe.tokenizer, TOK_CKPT_PATH)

    # Calibration + quantization (Text Encoder)
    mtq.quantize(pipe.text_encoder, quant_config, forward_encoder)

    print('>> DEBUG << \n\nPrinting text encoder quant summary:')
    mtq.print_quant_summary(pipe.text_encoder)
    print('\n\n')

    torch.save(mto.modelopt_state(pipe.text_encoder), ENC_QUANT_PATH)
    torch.save(pipe.text_encoder.state_dict(), ENC_SDICT_PATH)
    mto.save(pipe.text_encoder, ENC_CKPT_PATH)

    # Export
    #mto.restore(pipe.unet, CHECKPOINT_PATH) # TODO: Not needed?
    #quantize_lvl(pipe.unet, args.quant_level, nlayers)
    #mtq.disable_quantizer(pipe.unet, filter_func)
    #modelopt_export_sd(pipe.unet, ONNX_DIR)

    # QDQ needs to be in FP32
    #pipe.unet.to(torch.float32)
    #modelopt_export_sd(pipe, ONNX_DIR, MODEL_NAME)