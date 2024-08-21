import argparse
import os
import re
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# For storing and loading calibration data
import pickle

import torch
from diffusers import (
    AutoencoderTiny, LCMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
)
from diffusers.utils import load_image, make_image_grid
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from img_utils import downsample, tensor2img
from PIL import Image

from streamdiffusion import StreamUNetControlDiffusion, StreamUNetControlSeparatedDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import (
    accelerate_with_tensorrt_unetcontrol,
    accelerate_with_tensorrt_unet_separate_control
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.calib.max import MaxCalibrator

USE_PEFT = True
try:
    from peft.tuners.lora.layer import Conv2d as PEFTLoRAConv2d
    from peft.tuners.lora.layer import Linear as PEFTLoRALinear
except ModuleNotFoundError:
    USE_PEFT = False


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

# Args related to quantization
parser.add_argument('--ncal', type=int, default=1000)
parser.add_argument('--quant_encoder', type = eval, default = False, choices = [True, False])
parser.add_argument('--quant_controlnet', type = eval, default = False, choices = [True, False])
parser.add_argument('--quant_unet', type = eval, default = False, choices = [True, False])
parser.add_argument('--quant_decoder', type = eval, default = False, choices = [True, False])

parser.add_argument('--no_lcm', action = 'store_true')
parser.add_argument('--separate_controlnet', action = 'store_true')

parser.add_argument('--quant_level', type=float, default=3.0, choices=[1.0, 2.0, 2.5, 3.0])
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
# ======================= Functions ======================================
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

# ========================================================================
# ========================================================================
# ========================== Main = ======================================
# ========================================================================
# ========================================================================

if __name__ == "__main__":
    #################################################################
    # model loading
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11f1e_sd15_tile' if args.size == 512 else 'zjysteven/control_minisd_tile',
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        #"stabilityai/sdxl-turbo",
        "runwayml/stable-diffusion-v1-5" if args.size == 512 else "lambdalabs/miniSD-diffusers",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Enable acceleration
    #if args.accel == 'xformers':
    #    pipe.enable_xformers_memory_efficient_attention()
    #################################################################

    stream = StreamUNetControlSeparatedDiffusion(
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

    if args.no_lcm == False:
        # If the loaded model is not LCM, merge LCM
        stream.load_lcm_lora(
            "latent-consistency/lcm-lora-sdv1-5" if args.size == 512 else f"zjysteven/lcm-lora-miniSD-{args.lcm_id}"
        )
        stream.fuse_lora()
    
    # Load quantized model
    print(f'>>> Loading ./unet_quant/unet{args.num_layers}_quant* quantized model.')
    stream.pipe.unet = mto.restore_from_modelopt_state(pipe.unet, torch.load(f'./unet_quant/unet{args.num_layers}_quant_states_qint8.pt'))
    stream.pipe.unet.load_state_dict(torch.load(f'./unet_quant/unet{args.num_layers}_state_dict_qint8.pt'))

    # Debug - call quant_lvl
    nlayers = None if args.num_layers < 1 else args.num_layers
    quantize_lvl(stream.pipe.unet, args.quant_level, nlayers)

    print('>>> \n\nPrinting quantization summary:')
    mtq.print_quant_summary(stream.pipe.unet)
    print('\n\n')

    # Use Tiny VAE for further acceleration
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    # Load data
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

    # Save directory name based on quantization flags
    save_dir = f'./outputs/TRT_unet_quant_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_accel={args.accel}_framebff{args.frame_bff_size}_qlayers{args.num_layers}'
    os.makedirs(save_dir, exist_ok=True)

    input_buffer = []
    cond_buffer = []
    img_batch_size = stream.frame_bff_size
    num_batches = len(selected) // img_batch_size
    img_cnt = 0
    negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality, blur, blurred, DOF"] * img_batch_size

    # NOTE: TensorRT will need to be managed differently than in the previous pipeline
    if args.accel == 'trt':
        #mtq.disable_quantizer(stream.pipe.unet, filter_func)
        engine_dir = f"engines/quant_unet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}"
        stream = accelerate_with_tensorrt_unet_separate_control(
            stream,
            engine_dir,
            max_batch_size=stream.batch_size,
            engine_build_options={
                'opt_image_height': args.size,
                'opt_image_width': args.size,
                'quant_int8_modelopt': True,
            },
            quant_encoder = False, # No further quantization
            quant_controlnet = False,
            quant_unet = True,
            quant_decoder = False,
            encoder_calibration_loader = None,
            controlnet_calibration_loader = None,
            unet_calibration_loader = None,
            decoder_calibration_loader = None
        )
        """
        if args.separate_controlnet == True:
            if args.size == 256:
                engine_dir = f"engines/unet_separate_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}_lcm{args.lcm_id}"
            else:
                engine_dir = f"engines/unet_separate_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}"

            stream = accelerate_with_tensorrt_unet_separate_control(
                stream,
                engine_dir,
                max_batch_size = stream.batch_size,
                engine_build_options = {
                    'opt_image_height': args.size,
                    'opt_image_width': args.size
                },
                quant_encoder = args.quant_encoder,
                quant_controlnet = args.quant_controlnet,
                quant_unet = args.quant_unet,
                quant_decoder = args.quant_decoder,
                encoder_calibration_loader = enc_calib_list,
                controlnet_calibration_loader = cnet_calib_list,
                unet_calibration_loader = unet_calib_list,
                decoder_calibration_loader = dec_calib_list
            )
        else:
            if args.size == 256:
                engine_dir = f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}_lcm{args.lcm_id}"
            else:
                engine_dir = f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}"

            stream = accelerate_with_tensorrt_unetcontrol(
                stream,
                engine_dir,
                max_batch_size=stream.batch_size,
                engine_build_options={
                    'opt_image_height': args.size,
                    'opt_image_width': args.size
                },
                quant_encoder = args.quant_encoder,
                quant_unet = args.quant_unet,
                quant_decoder = args.quant_decoder,
                encoder_calibration_loader=enc_calib_list,
                unet_calibration_loader=unet_calib_list,
                decoder_calibration_loader=dec_calib_list
                # f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}", 
                # max_batch_size=stream.denoising_steps_num * 16, # max frame bff size is 128 on A100
            )
        """

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
        
        # ---- DEBUG ---- TODO
        #print(f'<< Profile script >> Calling stream()')
        images, _ = stream(cond_buffer[-img_batch_size:])

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

    with open(f'{save_dir}/ema_time.txt', 'a') as f:
        f.write('=========================================\n')
        f.write(f'Per batch: {stream.inference_time_ema:.6f} s\n')
        f.write(f'Per image: {stream.inference_time_ema/img_batch_size:.6f} s\n')
        f.write(f'FPS: {1/(stream.inference_time_ema/img_batch_size):.4f}\n')
        f.write('=========================================\n\n')