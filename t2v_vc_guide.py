from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from typing import Callable, List, Optional, Union
import sys
import os
from animatediff.models.unet import UNet3DConditionModel
from vc_utils.utils import instantiate_from_config
from filter_utils import (
    get_freq_filter,
    freq_mix_3d,
)
from types import SimpleNamespace
from scripts.evaluation.funcs import load_model_checkpoint
from utils import load_weights, save_videos_grid
from omegaconf import OmegaConf   
import numpy as np
import argparse 
import torch
from tqdm.auto import tqdm
from einops import rearrange
import time 
import PIL
import numpy as np
import inspect
import torch.nn.functional as F
from tqdm import trange
from torch.cuda.amp import autocast, GradScaler
import random

# Fix seed
def set_seed(seed: int):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
# Get x0 from noise
def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1) 
    pred_original_sample = (sample - (1 - alpha_t) ** (0.5) * model_output) / alpha_t ** (0.5)

    return pred_original_sample

class VideoGuidance:
    def __init__(self, args, device):
        super().__init__()
        # disable all gradient calculations
        torch.set_grad_enabled(False)
        
        if args.precision == "bfloat16":
            self.DTYPE = torch.bfloat16
        elif args.precision == "float16":
            self.DTYPE = torch.float16
        else:
            self.DTYPE = torch.float32
        self.device = device

        # initialize Videocrafter2 config
        model_config  = OmegaConf.load(args.config)[0]
        inference_config = OmegaConf.load(model_config.get("inference_config", None))
        vc_config = OmegaConf.load(args.vc_config)
        vc_model_config = vc_config.pop("model", OmegaConf.create())
        self.fps = args.fps

        self.tokenizer_one = CLIPTokenizer.from_pretrained(args.animatediff_model_path, subfolder="tokenizer")
        self.text_encoder_one = CLIPTextModel.from_pretrained(args.animatediff_model_path, subfolder="text_encoder", torch_dtype=self.DTYPE).to(device)
        self.vae          = AutoencoderKL.from_pretrained(args.animatediff_model_path, subfolder="vae", torch_dtype=self.DTYPE).to(device)

        self.vae_dtype = self.DTYPE

        # Initialize two models
        self.model = self.create_animatediff_model(args.animatediff_model_path, inference_config, model_config)
        self.model_guide = self.create_vc_model(args.vc_model_path, vc_model_config)
        self.model.enable_xformers_memory_efficient_attention()

        if args.precision == "float16":
            self.model.to(dtype=torch.float16)
          
        self.num_train_timesteps = args.num_train_timesteps
        self.vae_downsample_ratio = self.vae_scale_factor = 8

        # We use same scheduler for AnimateDiff and VideoCrafter2
        self.scheduler = DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # sampling parameters 
        self.num_step = args.num_step 
        self.ddpm_num_timesteps = 1000
        
        # Filter
        self.freq_filter = None
      
    def create_animatediff_model(self, model_path, inference_config, model_config):
      model = UNet3DConditionModel.from_pretrained_2d(model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(device=self.device)
      model = load_weights(
            model,
            self.vae,
            self.text_encoder_one,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to(self.device)
      
      return model

    @torch.no_grad()
    def init_filter(self, video_length, height, width, filter_params):
        # initialize frequency filter for noise reinitialization
        batch_size = 1
        num_channels_latents = self.model.in_channels
        filter_shape = [
            batch_size, 
            num_channels_latents, 
            video_length, 
            height // self.vae_scale_factor, 
            width // self.vae_scale_factor
        ]
        self.freq_filter = get_freq_filter(filter_shape, device=self.device, params=filter_params)
    
    def create_vc_model(self, model_path, config):
      model = instantiate_from_config(config)
      model = model.to(self.device)
      model = load_model_checkpoint(model, model_path)
      model.eval()     
      return model

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer_one( # text input
            prompt,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_one(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder_one.config, "use_attention_mask") and self.text_encoder_one.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder_one(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer_one(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder_one.config, "use_attention_mask") and self.text_encoder_one.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder_one( # What is text_encoder attention_mask
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def predict_guide_noise(self, zt, t, c, uc, **kwargs):
        noise_uc = self.model_guide.apply_model(zt, t, uc, **kwargs)
        noise_c = self.model_guide.apply_model(zt, t, c, **kwargs)
        return noise_uc, noise_c
    
    def encode_latents(self, latents):
        video_length = latents.shape[2]
        latents = (latents - 0.5) * 2
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.encode(latents[frame_idx:frame_idx+1]).latent_dist.sample())
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = video * 0.18215
        return video

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def sample_animatediff(self, noise, prompt_embed, cfg_scale=None, cfg_plus=False):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=self.DTYPE, device=self.device)

        ddim_timesteps = (
            np.linspace(0, self.num_train_timesteps-1, self.num_step+1)
            .round()
            .copy()
            .astype(np.int64)
        )
        all_timesteps = ddim_timesteps[1:][::-1]
        
        DTYPE = prompt_embed.dtype

        for i, constant in enumerate(tqdm(all_timesteps)):
    
            current_timestep = torch.ones(1, device=self.device, dtype=torch.long) * all_timesteps[i]
            if i == len(all_timesteps) - 1:
              next_timestep = torch.ones(1, device=self.device, dtype=torch.long) * 0
            else:
              next_timestep = torch.ones(1, device=self.device, dtype=torch.long) * all_timesteps[i+1]
              
            noise_input = torch.cat([noise] * 2)
            
            noise_s = self.model(
                noise_input, current_timestep, encoder_hidden_states=prompt_embed, is_video=True,
            ).sample
            
            noise_s_uc, noise_s_c = noise_s.chunk(2)
            noise_s_cfg = noise_s_uc + cfg_scale * (noise_s_c - noise_s_uc)
            
            init_eval_video = get_x0_from_noise(
                noise, noise_s_cfg, alphas_cumprod, current_timestep
            ).to(self.DTYPE)

            eval_video = init_eval_video
            alpha_t = alphas_cumprod[next_timestep]

            if cfg_plus:
                noise = alpha_t.sqrt() * eval_video + (1 - alpha_t).sqrt() * noise_s_uc
            else:
                noise = alpha_t.sqrt() * eval_video + (1 - alpha_t).sqrt() * noise_s_cfg

        eval_video = self.decode_latents(eval_video.to(self.vae_dtype))
        eval_video = torch.from_numpy(eval_video)

        return eval_video

    @torch.no_grad()
    def sample_video_guide(self, noise, prompt_embed, c_cond, uc_cond, use_vc, vc_extra_step=0, vc_interp_guide=0.0, vc_interp_step=0, cfg_plus=False, cfg_scale=None, **kwargs):

        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device, dtype=self.DTYPE)

        ddim_timesteps = (
            np.linspace(0, self.num_train_timesteps-1, self.num_step+1)
            .round()
            .copy()
            .astype(np.int64)
        )
        all_timesteps = ddim_timesteps[1:][::-1]
        DTYPE = prompt_embed.dtype

        for i, constant in enumerate(tqdm(all_timesteps)):
    
            current_timestep = torch.ones(1, device=self.device, dtype=torch.long) * all_timesteps[i]
            if i == len(all_timesteps) - 1:
                next_timestep = torch.ones(1, device=self.device, dtype=torch.long) * 0
            else:
                next_timestep = torch.ones(1, device=self.device, dtype=torch.long) * all_timesteps[i+1]
            
            noise_input = torch.cat([noise] * 2)
            
            noise_s = self.model(
                noise_input, current_timestep, encoder_hidden_states=prompt_embed,
            ).sample
            
            noise_s_uc, noise_s_c = noise_s.chunk(2)
            noise_s_cfg = noise_s_uc + cfg_scale * (noise_s_c - noise_s_uc)
            
            init_eval_video = get_x0_from_noise(
                noise, noise_s_cfg, alphas_cumprod, current_timestep
            ).to(self.DTYPE)

            # renoise into guidance space
            teacher_input_noise = self.scheduler.add_noise(
                init_eval_video.clone(), torch.randn_like(init_eval_video), current_timestep
            ).to(DTYPE)

            if i < vc_interp_step:
                new_ddim_timestep = (
                    np.linspace(0, constant, 25)
                    .round()
                    .copy()
                    .astype(np.int64)
                )
                all_new_timesteps = new_ddim_timestep[1:][::-1]
                for k, const in enumerate(all_new_timesteps[:vc_extra_step]):

                    current_new_timestep = torch.ones(1, device=self.device, dtype=torch.long) * all_new_timesteps[k]
                    if k == len(all_new_timesteps) - 1:
                        next_new_timestep = torch.ones(1, device=self.device, dtype=torch.long) * 0
                    else:
                        next_new_timestep = torch.ones(1, device=self.device, dtype=torch.long) * all_new_timesteps[k+1]

                    if not use_vc:
                        teacher_input_noise_input = torch.cat([teacher_input_noise] * 2)
                        noise_t = self.model(
                            teacher_input_noise_input, current_new_timestep, encoder_hidden_states=prompt_embed
                        ).sample
                        noise_uc_t, noise_c_t = noise_t.chunk(2)
                        noise_t = noise_uc_t + cfg_scale * (noise_c_t - noise_uc_t)
                    
                    else:
                        with autocast(dtype=torch.float16):
                            noise_uc_t, noise_c_t = self.predict_guide_noise(teacher_input_noise, current_new_timestep, c_cond, uc_cond, **kwargs)
                        noise_t = noise_uc_t + cfg_scale * (noise_c_t - noise_uc_t)
                        noise_t = noise_t.to(self.DTYPE)

                    guide_eval_video = get_x0_from_noise(
                        teacher_input_noise, noise_t, alphas_cumprod, current_new_timestep
                    ).to(self.DTYPE) # x0

                    alpha_t = alphas_cumprod[next_new_timestep]

                    if cfg_plus:
                        teacher_input_noise = alpha_t.sqrt() * guide_eval_video + (1 - alpha_t).sqrt() * noise_uc_t
                        save_videos_grid
                    else:
                        teacher_input_noise = alpha_t.sqrt() * guide_eval_video + (1 - alpha_t).sqrt() * noise_t
                
                eval_video = init_eval_video + vc_interp_guide * (guide_eval_video - init_eval_video)
       
            else:
                eval_video = init_eval_video.clone() 
            
            alpha_t = alphas_cumprod[next_timestep]

            if not cfg_plus:
                noise = alpha_t.sqrt() * eval_video + (1- alpha_t).sqrt() * noise_s_cfg
            else:
                noise = alpha_t.sqrt() * eval_video + (1- alpha_t).sqrt() * noise_s_uc

            if i < vc_interp_step:
                noise = noise.to(dtype=torch.float32)
                noise = freq_mix_3d(noise, torch.randn_like(noise), LPF=self.freq_filter).to(dtype=self.DTYPE)

        eval_video = self.decode_latents(eval_video.to(self.vae_dtype))
        eval_video = torch.from_numpy(eval_video)

        return eval_video
    
    @torch.no_grad()
    def inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        video_length: int,
        use_vc: bool,
        num_videos: int = 1,
        null_prompt: str = "",
        cfg_scale: float = 7.5,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        cfg_plus: bool = False,
        vc_extra_step: int = 0,
        vc_interp_guide: float = 0.0,
        vc_interp_step: int = 0,
        mode: int = 0,
    ):
 
        if seed == -1:
            seed = np.random.randint(0, 1000000)

        set_seed(seed)

        generator = torch.manual_seed(seed)

        noise = torch.randn(
            num_videos, 4, video_length, height // self.vae_downsample_ratio, width // self.vae_downsample_ratio, 
            generator=generator
        ).to(device=self.device, dtype=self.DTYPE)

        do_classifier_free_guidance = cfg_scale > 0.0

        text_embeddings = self._encode_prompt(
            prompt, self.device, num_videos_per_prompt, do_classifier_free_guidance, null_prompt
        ).to(dtype=self.DTYPE)
        
        c_emb = self.model_guide.get_learned_conditioning([prompt]).to(dtype=self.DTYPE)
        uc_emb  = self.model_guide.get_learned_conditioning([null_prompt]).to(dtype=self.DTYPE)
        c_cond = {'c_crossattn': [c_emb], 'fps': self.fps}
        uc_cond = {'c_crossattn': [uc_emb], 'fps': self.fps}
        kwargs = {'x0': None, 'temporal_length': video_length}

        self.init_filter(video_length=video_length, height=height, width=width, filter_params=SimpleNamespace(**{"method": "butterworth", "n": 4, "d_s": 0.125, "d_t": 0.125}))

        # Sampling for animatediff
        if mode == 0:
            sample_method = self.sample_animatediff
            eval_video = sample_method(
                noise=noise,
                prompt_embed=text_embeddings,
                cfg_scale=cfg_scale,
                cfg_plus=cfg_plus,
            )

        # Sampling for VideoGuide
        elif mode == 1:
            sample_method = self.sample_video_guide
            eval_video = sample_method(
                noise=noise,
                prompt_embed=text_embeddings,
                c_cond=c_cond,
                uc_cond=uc_cond,
                cfg_scale=cfg_scale,
                use_vc=use_vc,
                cfg_plus=cfg_plus,
                vc_extra_step=vc_extra_step,
                vc_interp_guide=vc_interp_guide,
                vc_interp_step=vc_interp_step,
                **kwargs
            )

        return eval_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--animatediff_model_path", type=str, required=True)
    parser.add_argument("--vc_model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vc_config", type=str, required=True)
    parser.add_argument("--precision", type=str, default="float16", choices=["float32", "float16"])
    parser.add_argument("--num_step", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="A drone view of celebration with Christmas tree and fireworks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--savedir", type=str, default="vbench_result")
    parser.add_argument("--cfg_plus", action="store_true")

    # guidance argument
    parser.add_argument("--mode", type=int, default=1, help="0 for sampling AnimateDiff, 1 for VideoGuide")
    parser.add_argument("--vc_extra_step", type=int, default=10, help="Number of steps for extra denoising = Tau")
    parser.add_argument("--vc_interp_guide", type=float, default=0.2, help="Interpolation coefficient = 1 - Beta")
    parser.add_argument("--vc_interp_step", type=int, default=5, help="Number of guiding steps I for interpolating")
    parser.add_argument("--use_vc", type=bool, default=True, help="Use videocrafter as guidance model")


    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = VideoGuidance(args, device)

    method = ["AnimateDiff", "VideoGuide"]

    with open(args.prompt, "r") as f:
        prompt_list = f.read().splitlines()
    
    for prompt in prompt_list:

        output = model.inference(
            prompt=prompt, seed=args.seed, height=args.image_resolution, width=args.image_resolution, video_length=args.video_length,
            cfg_scale=args.cfg_scale, cfg_plus=args.cfg_plus, vc_extra_step=args.vc_extra_step, vc_interp_guide=args.vc_interp_guide, vc_interp_step=args.vc_interp_step, use_vc=args.use_vc, 
            mode=args.mode
        )

        video = output
        method_name = method[args.mode]
        save_videos_grid(video, f"./result/{prompt}/{method_name}.gif")
        
if __name__ == "__main__":
    main()
