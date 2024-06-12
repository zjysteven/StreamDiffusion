import argparse
import os
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

# Args related to quantization
parser.add_argument('--ncal', type=int, default=1000)
parser.add_argument('--quant_encoder', type = eval, default = False, choices = [True, False])
parser.add_argument('--quant_unet', type = eval, default = False, choices = [True, False])
parser.add_argument('--quant_decoder', type = eval, default = False, choices = [True, False])

parser.add_argument('--no_lcm', action = 'store_true')

args = parser.parse_args()

# ========================================================================
# ========================================================================
# ======================= Functions ======================================
# ========================================================================
# ========================================================================

"""
Build calibration dictionaries for encoder, unet, and decoder engines.
Calibration requires representative data, and the stream has a coupled
processing chain of encoder --> unet --> decoder, along with intermediate
pre- and post-processing steps. We thus need to recreate this process to
produce good representative data.

NVIDIA documentation indicates that ~500 images are sufficient for calibrating
ImageNet classification networks. This is a more difficult scenario, and so
we will select 1000 images from the complete set.

NOTE: Many variables used in this function are declared in __main__, so this
      can only be called following those declarations.
"""
def load_quant_calib_data():
    enc_calib_data = None
    unet_calib_data = None
    dec_calib_data = None

    # Generate data, or load if cached
    enc_calib_list = []
    unet_calib_list = []
    dec_calib_list = []

    if args.no_lcm == True:
        calib_dat_fname = f'calib_data{args.size}_{args.num_inference_steps}steps_nolcm.pickle'
    elif args.size == 256:
        calib_dat_fname = f'calib_data{args.size}_{args.num_inference_steps}steps_lcm{args.lcm_id}.pickle'
    else:
        calib_dat_fname = f'calib_data{args.size}_{args.num_inference_steps}steps.pickle'

    if os.path.exists(calib_dat_fname):
        print('>>> Loading existing calibration data.')
        calib_file = open(calib_dat_fname, 'rb')
        calib_data = pickle.load(calib_file)
        enc_calib_list = calib_data['encoder_data']
        unet_calib_list = calib_data['unet_data']
        dec_calib_list = calib_data['decoder_data']
        calib_file.close()
    else:
        # Build calibrator dictionaries with real representative data
        print(f'>>> Generating calibrator dictionaries for {args.ncal} images.')
        for i in tqdm(range(min(len(selected), args.ncal))):
            filename = selected[i]

            prompt = [f'{img2caps[filename][0]}, realistic, best quality, extremely detailed']
            original = load_image(os.path.join(data_path, filename)).resize((args.size, args.size))
            cond = downsample(original, (args.cond_size, args.cond_size))
            cond_resized = cond.resize((args.size, args.size))

            if i == 0:
                stream.prepare(prompt, negative_prompt, guidance_scale = 1.2)
            else:
                stream.update_prompt(prompt)

            # Generate calibration dictionary items for image
            enc_item, unet_item, dec_item = stream.gen_calib_dict_intermediates(cond_resized)

            # Append to feed dictionary lists
            enc_calib_list.append(enc_item)
            unet_calib_list.append(unet_item)
            dec_calib_list.append(dec_item)

        # Save calibration data to a .pickle file for future use
        calib_data = {'encoder_data': enc_calib_list, 'unet_data': unet_calib_list, 'decoder_data': dec_calib_list}
        calib_file = open(calib_dat_fname, 'wb')
        pickle.dump(calib_data, calib_file)
        calib_file.close()

    # Format and set outputs if flag is set
    if args.quant_encoder == True:
        enc_calib_data = []
        for i in range(int(len(enc_calib_list) / 4)):
            calib_dat_list = []
            for item in enc_calib_list[i*4:(i*4)+4]:
                calib_dat_list.append(np.squeeze(item["images"]))
            enc_calib_data.append({"images": np.array(calib_dat_list)})

    if args.quant_unet == True:
        # NOTE: For some reason, the UNet data shape is fine.. this is probabily to
        #       do with the internals of the TensorRT usage in StreamDiffusion.
        unet_calib_data = unet_calib_list

    if args.quant_decoder == True:
        dec_calib_data = []
        for i in range(int(len(dec_calib_list) / 4)):
            calib_dat_list = []
            for item in dec_calib_list[i*4:(i*4)+4]:
                calib_dat_list.append(np.squeeze(item["latent"]))
            dec_calib_data.append({"latent": np.array(calib_dat_list)})

    return enc_calib_data, unet_calib_data, dec_calib_data

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

    if args.no_lcm == False:
        # If the loaded model is not LCM, merge LCM
        stream.load_lcm_lora(
            "latent-consistency/lcm-lora-sdv1-5" if args.size == 512 else f"zjysteven/lcm-lora-miniSD-{args.lcm_id}"
        )
        stream.fuse_lora()

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
    save_dir = f'./outputs/controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_accel={args.accel}_framebff{args.frame_bff_size}'
    if args.quant_encoder == True:
        save_dir += '_encoderQuant=INT8'
    if args.quant_unet == True:
        save_dir += '_UNetQuant=INT8'
    if args.quant_decoder == True:
        save_dir += '_decoderQuant=INT8'
    if args.no_lcm == True:
        save_dir += '_no_LCM_LoRA'

    os.makedirs(save_dir, exist_ok=True)

    input_buffer = []
    cond_buffer = []
    img_batch_size = stream.frame_bff_size
    num_batches = len(selected) // img_batch_size
    img_cnt = 0
    negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality, blur, blurred, DOF"] * img_batch_size

    if args.accel == 'trt':
        # List will be 'None' if flag is not set to quantize the net
        enc_calib_list, unet_calib_list, dec_calib_list = load_quant_calib_data()

        if args.no_lcm == True:
            engine_dir = f'engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}_nolcm'
        elif args.size == 256:
            engine_dir = f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}_lcm{args.lcm_id}"
        else:
            engine_dir = f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}"

        stream = accelerate_with_tensorrt_unetcontrol(
            stream,
            engine_dir,
            #f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}" \
            #if args.size != 256 else \
            #f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}_lcm{args.lcm_id}",
            max_batch_size=stream.batch_size,
            engine_build_options={
                'opt_image_height': args.size,
                'opt_image_width': args.size
            },
            #engine_dir_quant = eng_dir_quant,
            quant_encoder = args.quant_encoder,
            quant_unet = args.quant_unet,
            quant_decoder = args.quant_decoder,
            encoder_calibration_loader=enc_calib_list,
            unet_calibration_loader=unet_calib_list,
            decoder_calibration_loader=dec_calib_list
            # f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}", 
            # max_batch_size=stream.denoising_steps_num * 16, # max frame bff size is 128 on A100
        )

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