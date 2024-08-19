import gradio as gr
from gradio_log import Log

import os
import sys
from typing import Literal, Dict, Optional, Union
import logging
logging.getLogger('gradio').setLevel(logging.ERROR)
logging.getLogger('gradio_log').setLevel(logging.ERROR)

import torch
from torchvision.io import read_video, write_video
from torchvision.transforms import functional as TF
from tqdm import tqdm
from vidgear.gears import CamGear

from diffusers import (
    AutoencoderTiny, LCMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
)

from streamdiffusion import StreamUNetControlDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt_unetcontrol


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(CURRENT_DIR, "..", "demo_files")
VIDEO_DIR = os.path.join(DEMO_DIR, "video_outputs")
if os.path.isfile(os.path.join(VIDEO_DIR, "output.mp4")):
    os.remove(os.path.join(VIDEO_DIR, "output.mp4"))
os.makedirs(VIDEO_DIR, exist_ok=True)

log_file = os.path.join(DEMO_DIR, "log.txt")
logging.basicConfig(filename=log_file,
                    filemode='w',
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


##########################################################################
# loading model
logging.info("Model is being loaded...")

size = 256
accel = 'xformers'

controlnet = ControlNetModel.from_pretrained(
    'lllyasviel/control_v11f1e_sd15_tile' if size == 512 else 'zjysteven/control_minisd_tile',
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5" if size == 512 else "lambdalabs/miniSD-diffusers",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

if accel == 'xformers':
    pipe.enable_xformers_memory_efficient_attention()

frame_buffer_size = 30 if size == 256 else 15
stream = StreamUNetControlDiffusion(
    pipe,
    height=size,
    width=size,
    num_inference_steps=4,
    strength=1.0,
    torch_dtype=torch.float16,
    cfg_type="none",
    frame_buffer_size=frame_buffer_size,
)
# stream.enable_similar_image_filter(
#     0.90, # similar_image_filter_threshold,
#     15, # similar_image_filter_max_skip_frame,
# )
delay = stream.denoising_steps_num
stream.load_lcm_lora("zjysteven/lcm-lora-miniSD-1e-6")
stream.fuse_lora()
# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

if accel == 'trt':
    stream = accelerate_with_tensorrt_unetcontrol(
        stream,
        f"{CURRENT_DIR}/../engines/unet_controlnet_size{size}-64_steps4_cfg=none_framebff{frame_buffer_size}",
        max_batch_size=stream.batch_size,
        engine_build_options={
            'opt_image_height': size,
            'opt_image_width': size, 
        }
        # f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}", 
        # max_batch_size=stream.denoising_steps_num * 16, # max frame bff size is 128 on A100
    )

logging.info("Model has been successfully loaded!")
##########################################################################


def main(
    video_url: Optional[Union[str, None]],
    video_file: Optional[Union[str, None]],
    prompt: str,
):
    ##################################################################################################
    # video loading and preprocessing
    if video_url and video_file is not None:
        logging.error("Please provide a single input via either `Video URL' or `Video'!")
        yield None

    input_buffer = []
    if video_url:
        options = {
            "CAP_PROP_FPS": 30,
            "STREAM_RESOLUTION": "240p",
        }
        video_stream = CamGear(
            source=video_url,
            colorspace="COLOR_BGR2RGB",
            stream_mode=True, logging=False, **options
        ).start()
        print("video stream framerate: ", video_stream.framerate)

        while True:
            frame = video_stream.read()
            if frame is None:
                break
            input_buffer.append(torch.from_numpy(frame) / 255)
        video_stream.stop()
    else:
        video_input = read_video(video_file)
        video = video_input[0] / 255
        input_buffer.extend(list(video))
    
    print("len input_buffer: ", len(input_buffer))
    if len(input_buffer) < frame_buffer_size:
        logger.error("The input video is too short. Please provide a longer one.")
    input_buffer = input_buffer[:(len(input_buffer) // frame_buffer_size) * frame_buffer_size]
    num_frames = len(input_buffer)
    print("num_frames: ", num_frames)
    for i in range(num_frames):
        input_buffer[i] = TF.resize(
            input_buffer[i].permute(2, 0, 1), (size, size)
        ).permute(1, 2, 0)
    
    # for display purpose
    cache_input_buffer = torch.stack(input_buffer) * 255
    write_video(
        os.path.join(VIDEO_DIR, "input.mp4"), 
        cache_input_buffer,
        fps=30
    )

    input_buffer = input_buffer + [input_buffer[-1]] * (delay - 1) * frame_buffer_size
    num_batches = len(input_buffer) // frame_buffer_size

    logger.info("Video is loaded!")
    ##################################################################################################

    if not prompt:
        logging.error("Please provide a prompt!")
    
    stream.prepare(
        [prompt] * frame_buffer_size,
        guidance_scale=1.0
    )

    video_result = torch.zeros(num_frames, size, size, 3)
    # encoded_buffer = []
    frame_cnt = 0
    for i in tqdm(range(num_batches), total=num_batches):
        frames = input_buffer[i*frame_buffer_size:(i+1)*frame_buffer_size]
        encoded_buffer = TF.resize(
            torch.stack(frames).permute(0, 3, 1, 2), (64, 64)
        )
        
        output_frames = stream(encoded_buffer)

        if i >= delay - 1:
            for j in range(len(output_frames)):
                video_result[frame_cnt] = postprocess_image(
                    output_frames[j].permute(1, 2, 0),
                    output_type="pt"
                )
                yield TF.to_pil_image(video_result[frame_cnt].permute(2, 0, 1))
                frame_cnt += 1
                
    video_result = video_result * 255
    boundary = 0.5 * torch.ones(num_frames, size, 10, 3)
    write_video(
        os.path.join(VIDEO_DIR, "output.mp4"), 
        torch.cat([cache_input_buffer, boundary, video_result], dim=2),
        fps=30
    )

    logger.info("Decoded video is saved!")
    logger.info(f"The decoding speed is {1/(stream.inference_time_ema/frame_buffer_size):.1f} FPS")
    logger.info("=" * 50)


def play_video():
    return os.path.join(VIDEO_DIR, "output.mp4")


with gr.Blocks(
    title="Video Compression",
    # css="#gradio-log-comp-id {min-height: 100px; max-height: 250px}"
) as demo:
    gr.Markdown(
    """
    # Video Compression Demo
    Follow these steps:
    1. To provide your input video, ***either*** specify a URL (e.g. from YouTube) in the `Video URL` block, ***or*** upload a video file (or using webcam) in the `Video Input` block. **Choose only one and do not do both.** Also, make sure your video has **at least 30 frames (at least a few seconds long)**.
    2. Provide a description of the video content in `Prompt`. It cannot be empty.
    3. Click the `Run` button, and the program will encode the input video and decode it in a real-time fashion. The decoded frames will be shown in `Real-Time Frame Output`.
    4. Lastly you can see a comparison between the original and decoded video by clicking `Play Decoded Video` button.
    """
    )

    # gr.HTML(
    # '''<html>
    # <body>

    #     <h1>My First JavaScript</h1>

    # </body>
    # </html>
    # '''
    # )

    with gr.Row():
        video_url = gr.Textbox(label="Video URL", show_copy_button=True)
        prompt_input = gr.Textbox(label="Prompt")
    
    with gr.Row():  
        video_file = gr.Video(
            sources=["upload", "webcam"], scale=1,
            # height=512, width=512,
            label="Video Input",
            mirror_webcam=False
        )

        real_time_output = gr.Image(
            label="Real-Time Frame Output", scale=1
            # height=512, width=512
        )

        video_output = gr.Video(label="Video Output", scale=2)
        # video_output.render()

    with gr.Row():
        submit_btn = gr.Button("Run", scale=1)
        submit_btn.click(
            fn=main, 
            inputs=[video_url, video_file, prompt_input], 
            outputs=real_time_output
        )

        play_btn = gr.Button("Play Decoded Video", scale=1)
        play_btn.click(
            fn=play_video,
            outputs=video_output
        )

    gr.Examples(
        [
            ["https://youtu.be/geNCpS885tg?si=B5OLbSyEzBHjShDg", "a man washing a car, cartoon, animation"],
            ["https://youtu.be/xuP4g7IDgDM?si=LYOt1xmuOrGvOMUB", "stunning sunset seen from the sea, realistic, natural"],
        ],
        [video_url, prompt_input],
    )

    Log(log_file, dark=True, xterm_font_size=16, elem_id="gradio-log-comp-id")

demo.launch(share=True)