CUDA_VISIBLE_DEVICES=7 \
python profile_controlnet.py \
    --accel trt \
    --num_inference_steps 4 \
    --strength 0.8 \
    --cfg_type none \
    --size 512 --scale 8