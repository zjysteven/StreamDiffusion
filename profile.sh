for fbs in 1; do
    # Baseline
    CUDA_VISIBLE_DEVICES=5 \
    python profile_controlnet.py \
        --accel trt \
        --num_inference_steps 4 \
        --strength 1.0 \
        --cfg_type none \
        --size 512 --cond_size 64 \
        --ncal 1000 \
        --quant_encoder False \
        --quant_controlnet False \
        --quant_unet True \
        --quant_decoder False \
        --separate_controlnet \
        --frame_bff_size ${fbs}
done