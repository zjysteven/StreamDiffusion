# StreamDiffusion with ControlNet Integration

For original README, please refer to [README_original.md](./README_original.md).

## Setup

```
conda env create -f environment.yml
conda activate vc
cd streamdiffusion
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
python -m pip install av
cd ../diffusers
python -m pip install -e .
```

## Running examples

- profiling controlnet for image generation
```bash
for fbs in 1; do
    CUDA_VISIBLE_DEVICES=1 \
    python scripts/profile_controlnet.py \
        --accel trt \
        --num_inference_steps 4 \
        --strength 1.0 \
        --cfg_type none \
        --size 256 --cond_size 64 \
        --frame_bff_size ${fbs}
done
```

- video demo
```bash
CUDA_VISIBLE_DEVICES=3 \
python scripts/video_demo.py \
    --video_path test.mp4 \
    --prompt "dog running" \
    --size 256
```

- online video demo
```bash
# "https://youtu.be/xuP4g7IDgDM?si=LYOt1xmuOrGvOMUB"
# "stunning sunset seen from the sea"

CUDA_VISIBLE_DEVICES=3 \
python scripts/video_stream_yt_demo.py \
    --video_url "https://youtu.be/geNCpS885tg?si=B5OLbSyEzBHjShDg" \
    --prompt "a man washing a car, cartoon, animation" \
    --size 256
```