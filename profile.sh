for fbs in 1; do
    CUDA_VISIBLE_DEVICES=6 \
    python profile_controlnet.py \
        --accel trt \
        --num_inference_steps 50 \
        --strength 1.0 \
        --cfg_type none \
        --size 256 --cond_size 64 \
        --ncal 1000 \
        --quant_encoder False \
        --quant_unet True \
        --quant_decoder False \
        --frame_bff_size ${fbs}
done