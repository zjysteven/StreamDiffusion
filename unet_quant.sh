GPU=1
NSTEPS=4
OSIZE=512
CNDSIZE=64
CALSIZE=512
QLVL=3.0
ALPH=1.5
PRCNT=1.0
NIMG=50000

for nlayers in 0; do
    # Quantization
    CUDA_VISIBLE_DEVICES=$GPU \
    python sd_quant.py \
        --num_inference_steps $NSTEPS \
        --size $OSIZE \
        --cond_size $CNDSIZE \
        --calib_size $CALSIZE \
        --quant_level $QLVL \
        --alpha $ALPH \
        --percentile $PRCNT \
        --num_layers $nlayers

    # Profiling
    #CUDA_VISIBLE_DEVICES=$GPU \
    #python profile_quant.py \
    #    --num_inference_steps $NSTEPS \
    #    --accel trt \
    #    --strength 1.0 \
    #    --cfg_type none \
    #    --size $OSIZE \
    #    --cond_size $CNDSIZE \
    #    --num_images $NIMG \
    #    --quant_level $QLVL \
    #    --num_layers $nlayers \
    #    --frame_bff_size 1
done