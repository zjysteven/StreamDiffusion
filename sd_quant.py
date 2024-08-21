"""
Quantization of SD-1.5 + LCM-LoRA directly through TensorRT-Model-Optimizer package.
Provides greater fidelity and control of quantization effort compared to StreamDiffusion project.

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
# =========================== Constants ==================================
# ========================================================================
# ========================================================================
AXES_NAME = {
    "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    "timestep": {0: "steps"},
    "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
    "down_block_res_samples_0": {0: "batch_size", 1: "HD", 2: "height", 3: "weight"},
    "down_block_res_samples_1": {0: "batch_size", 1: "HD", 2: "H", 3: "W"},
    "down_block_res_samples_2": {0: "batch_size", 1: "HD", 2: "H", 3: "W"},
    "down_block_res_samples_3": {0: "batch_size", 1: "HD", 2: "H2", 3: "W2"},
    "down_block_res_samples_4": {0: "batch_size", 1: "2HD", 2: "H2", 3: "W2"},
    "down_block_res_samples_5": {0: "batch_size", 1: "2HD", 2: "H2", 3: "W2"},
    "down_block_res_samples_6": {0: "batch_size", 1: "2HD", 2: "H4", 3: "W4"},
    "down_block_res_samples_7": {0: "batch_size", 1: "4HD", 2: "H4", 3: "W4"},
    "down_block_res_samples_8": {0: "batch_size", 1: "4HD", 2: "H4", 3: "W4"},
    "down_block_res_samples_9": {0: "batch_size", 1: "4HD", 2: "H8", 3: "W8"},
    "down_block_res_samples_10": {0: "batch_size", 1: "4HD", 2: "H8", 3: "W8"},
    "down_block_res_samples_11": {0: "batch_size", 1: "4HD", 2: "H8", 3: "W8"},
    "mid_block_res_sample": {0: "batch_size", 1: "4HD", 2: "H8", 3: "W8"},
    "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
}

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
parser.add_argument('--num_layers', type=int, default=1)
args = parser.parse_args()

# ========================================================================
# ========================================================================
# ========================== Calibrator ==================================
# ========================================================================
# ========================================================================
class PercentileCalibrator(MaxCalibrator):
    def __init__(self, num_bits=8, axis=None, unsigned=False, track_amax=False, **kwargs):
        super().__init__(num_bits, axis, unsigned, track_amax)
        self.percentile = kwargs['percentile']
        self.total_step = kwargs['total_step']
        self.collect_method = kwargs['collect_method']
        self.data = {}
        self.i = 0

    def collect(self, x):
        """Tracks the absolute max of all tensors.

        Args:
            x: A tensor

        Raises:
            RuntimeError: If amax shape changes
        """
        # Swap axis to reduce.
        axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
        # Handle negative axis.
        axis = [x.dim() + i if isinstance(i, int) and i < 0 else i for i in axis]
        reduce_axis = []
        for i in range(x.dim()):
            if i not in axis:
                reduce_axis.append(i)
        local_amax = mtq.utils.reduce_amax(x, axis=reduce_axis).detach()
        _cur_step = self.i % self.total_step
        if _cur_step not in self.data.keys():
            self.data[_cur_step] = local_amax
        else:
            if self.collect_method == "global_min":
                self.data[_cur_step] = torch.min(self.data[_cur_step], local_amax)
            elif self.collect_method == "min-max" or self.collect_method == "mean-max":
                self.data[_cur_step] = torch.max(self.data[_cur_step], local_amax)
            else:
                self.data[_cur_step] += local_amax
        if self._track_amax:
            raise NotImplementedError
        self.i += 1

    def compute_amax(self):
        """Return the absolute max of all tensors collected."""
        up_lim = int(self.total_step * self.percentile)
        if self.collect_method == "min-mean":
            amaxs_values = [self.data[i] / self.total_step for i in range(0, up_lim)]
        else:
            amaxs_values = [self.data[i] for i in range(0, up_lim)]
        if self.collect_method == "mean-max":
            act_amax = torch.vstack(amaxs_values).mean(axis=0)[0]
        else:
            act_amax = torch.vstack(amaxs_values).min(axis=0)[0]
        self._calib_amax = act_amax
        return self._calib_amax

    def __str__(self):
        s = "PercentileCalibrator"
        return s.format(**self.__dict__)

    def __repr__(self):
        s = "PercentileCalibrator("
        s += super(MaxCalibrator, self).__repr__()
        s += " calib_amax={_calib_amax}"
        if self._track_amax:
            s += " amaxs={_amaxs}"
        s += ")"
        return s.format(**self.__dict__)


# ========================================================================
# ========================================================================
# =========================== Utilities ==================================
# ========================================================================
# ========================================================================
def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding).*"
    )
    return pattern.match(name) is not None


def quantize_lvl(unet, quant_level=2.5, num_layers = None):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration

    NOTE: Modify this function to select layer(s) to quantize
    """
    if num_layers is None:
        qmax = float('inf')
    else:
        qmax = num_layers

    qn = 0
    for name, module in unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            if qn < qmax:
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
                qn += 1
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                if qn < qmax:
                    module.input_quantizer.enable()
                    module.weight_quantizer.enable()
                    qn += 1
                else:
                    module.input_quantizer.disable()
                    module.weight_quantizer.disable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
                

# NOTE: Modify this function to select layer(s) to quantize
def get_int8_config(
    model,
    quant_level=3,
    alpha=0.8,
    percentile=1.0,
    num_inference_steps=20,
    collect_method='min-mean',
    num_layers = None,
):
    # Quantize up to num_layers. If None, quantize all layers
    if num_layers is None:
        qmax = float('inf')
    else:
        qmax = num_layers

    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }

    qn = 0
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                if qn < qmax:
                    print(f'>> DEBUG << Adding layer {name} to quantization config.')
                    quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                    quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
                    qn += 1
        elif isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            if qn < qmax:
                print(f'>> DEBUG << Adding layer {name} to quantization config.')
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
                """
                quant_config["quant_cfg"][i_name] = {
                    "num_bits": 8,
                    "axis": None,
                    "calibrator": calibrator,
                    "calibrator": (
                        PercentileCalibrator,
                        (),
                        {
                            "num_bits": 8,
                            "axis": None,
                            "percentile": percentile,
                            "total_step": num_inference_steps,
                            "collect_method": collect_method,
                        },
                    ),
                }
                """
                qn += 1

    return quant_config


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


def generate_dummy_inputs(device):
    _BS = 4   # Batch Size
    _NC = 4   # Num Channels
    _LH = 64  # Latent Height
    _LW = 64  # Latent Width
    _SL = 77  # Sequence Length
    _ED = 768 # Embedding Dimension
    _HD = 320 # Hidden Dimension
    _TS = 4   # Time Steps
    dummy_input = {}
    dummy_input['sample'] = torch.ones(_BS, _NC, _LH, _LW).to(device).half()
    dummy_input['timestep'] = torch.ones(_TS).to(device).half()
    dummy_input['encoder_hidden_states'] = torch.ones(_BS, _SL, _ED).to(device).half()
    dummy_input['down_block_res_samples_0'] = torch.ones(_BS, _HD, _LH, _LW).to(device).half()
    dummy_input['down_block_res_samples_1'] = torch.ones(_BS, _HD, _LH, _LW).to(device).half()
    dummy_input['down_block_res_samples_2'] = torch.ones(_BS, _HD, _LH, _LW).to(device).half()
    dummy_input['down_block_res_samples_3'] = torch.ones(_BS, _HD, _LH // 2, _LW // 2).to(device).half()
    dummy_input['down_block_res_samples_4'] = torch.ones(_BS, _HD * 2, _LH // 2, _LW // 2).to(device).half()
    dummy_input['down_block_res_samples_5'] = torch.ones(_BS, _HD * 2, _LH // 2, _LW // 2).to(device).half()
    dummy_input['down_block_res_samples_6'] = torch.ones(_BS, _HD * 2, _LH // 4, _LW // 4).to(device).half()
    dummy_input['down_block_res_samples_7'] = torch.ones(_BS, _HD * 4, _LH // 4, _LW // 4).to(device).half()
    dummy_input['down_block_res_samples_8'] = torch.ones(_BS, _HD * 4, _LH // 4, _LW // 4).to(device).half()
    dummy_input['down_block_res_samples_9'] = torch.ones(_BS, _HD * 4, _LH // 8, _LW // 8).to(device).half()
    dummy_input['down_block_res_samples_10'] = torch.ones(_BS, _HD * 4, _LH // 8, _LW // 8).to(device).half()
    dummy_input['down_block_res_samples_11'] = torch.ones(_BS, _HD * 4, _LH // 8, _LW // 8).to(device).half()
    dummy_input['mid_block_res_sample'] = torch.ones(_BS, _HD * 4, _LH // 8, _LW // 8).to(device).half()

    return dummy_input


def modelopt_export_sd(model, onnx_dir):
    os.makedirs(f'{onnx_dir}', exist_ok=True)
    dummy_inputs = generate_dummy_inputs(device = model.device)

    output = Path(f'{onnx_dir}/unet_quantINT8.onnx')
    input_names = [
        'sample', 'timestep', 'encoder_hidden_states',
        'down_block_res_samples_0', 'down_block_res_samples_1', 'down_block_res_samples_2', 'down_block_res_samples_3',
        'down_block_res_samples_4', 'down_block_res_samples_5', 'down_block_res_samples_6', 'down_block_res_samples_7',
        'down_block_res_samples_8', 'down_block_res_samples_9', 'down_block_res_samples_10', 'down_block_res_samples_11',
        'mid_block_res_sample'
    ]
    output_names = ['latent']

    dynamic_axes = AXES_NAME
    do_constant_folding = True
    opset_version = 17

    onnx_export(
        model,
        (dummy_inputs,),
        f=output.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(str(output), load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(str(output), load_external_data=True)
        onnx.save(
            onnx_model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + "_data",
            size_threshold=1024,
        )
        for tensor in tensors_paths:
            os.remove(output.parent / tensor)


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


# ========================================================================
# ========================================================================
# =========================== Main Func ==================================
# ========================================================================
# ========================================================================
if __name__ == "__main__":
    # Some constants..
    ONNX_DIR = f'./engines/unet512_{args.num_layers}_layers/onnx'
    QUANT_DIR = f'./unet_quant'
    MODEL_CKPT_PATH = os.path.join(QUANT_DIR, f'unet{args.num_layers}_state_dict_qint8.pt')
    QUANT_CKPT_PATH = os.path.join(QUANT_DIR, f'unet{args.num_layers}_quant_states_qint8.pt')
    CHECKPOINT_PATH = os.path.join(QUANT_DIR, 'unet_ckpt')
    MODEL_NAME = 'runwayml/stable-diffusion-v1-5'

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

    nlayers = None if args.num_layers < 1 else args.num_layers

    # Get quant configuration
    quant_config = get_int8_config(
        pipe.unet,
        args.quant_level,
        args.alpha,
        args.percentile,
        args.num_inference_steps + extra_step,
        collect_method='global_min',
        num_layers = nlayers,
    )

    # Define forward loop for calibration
    def forward_loop(unet):
        pipe.unet = unet
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

    # Calibration + quantization
    mtq.quantize(pipe.unet, quant_config, forward_loop)

    print('>> DEBUG << \n\nPrinting quantization summary:')
    mtq.print_quant_summary(pipe.unet)
    print('\n\n')

    torch.save(mto.modelopt_state(pipe.unet), QUANT_CKPT_PATH)
    torch.save(pipe.unet.state_dict(), MODEL_CKPT_PATH)
    mto.save(pipe.unet, CHECKPOINT_PATH)

    # Export
    #mto.restore(pipe.unet, CHECKPOINT_PATH) # TODO: Not needed?
    quantize_lvl(pipe.unet, args.quant_level, nlayers)
    mtq.disable_quantizer(pipe.unet, filter_func)
    modelopt_export_sd(pipe.unet, ONNX_DIR)

    # QDQ needs to be in FP32
    #pipe.unet.to(torch.float32)
    #modelopt_export_sd(pipe, ONNX_DIR, MODEL_NAME)