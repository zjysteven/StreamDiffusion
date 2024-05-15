for fbs in 1; do
    CUDA_VISIBLE_DEVICES=1 \
    python profile_controlnet.py \
        --accel trt \
        --num_inference_steps 4 \
        --strength 1.0 \
        --cfg_type none \
        --size 256 --cond_size 64 \
        --frame_bff_size ${fbs}
done