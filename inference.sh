name="videoguide"

animatediff_ckpt='../animatediff/models/StableDiffusion/stable-diffusion-v1-5' # Change into your stable-diffusion-v1-5 path
ckpt='../fifo-diffusion/checkpoints/videocrafter/base_512_v2/model.ckpt' # Change into your videocrafter path
# config='./animatediff_configs/prompts/v2/v2-1-Film.yaml'
# config='./animatediff_configs/prompts/v2/v2-1-ToonYou.yaml'
# config='./animatediff_configs/prompts/v2/v2-1-RealisticVision.yaml'
config='./animatediff_configs/prompts/v2/v2-1-w-dreambooth.yaml'
vc_config='./vc_configs/inference_t2v_512_v2.0.yaml'

prompts="./prompts/prompt.txt"

python t2v_vc_guide.py \
    --seed 42 \
    --video_length 16 \
    --fps 8 \
    --cfg_scale 0.8 \
    --animatediff_model_path $animatediff_ckpt \
    --vc_model_path $ckpt \
    --config $config \
    --vc_config $vc_config \
    --num_step 50 \
    --savedir "./result" \
    --precision 'float16' \
    --prompt "$prompts" \
    --mode 1 \
    --cfg_plus


