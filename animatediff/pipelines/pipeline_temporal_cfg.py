# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange
from animatediff.utils.util import save_videos_grid
from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb
import math
import os 

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def lamb_schedule(start, end, step, device, type='linear'):
    if type == 'linear':
        a = torch.linspace(start, end, step, device=device)
        return a
        
        
    elif type == 'log_linear':
        start_log = math.log(start)
        end_log = math.log(end)
        a = torch.linspace(start_log, end_log, step, device=device)
        a = torch.exp(a)
        return a
    

@dataclass
class DDIMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class TemporalCFGPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[SparseControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer( # text input
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
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
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder( # What is text_encoder attention_mask
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

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video
      

    # extra_step_kwargs['eta'], extra_step_kwargs['generator']
    def prepare_extra_step_kwargs(self, generator, eta): # eta and corresponding generator 설정 in DDIM
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

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # Return Initial random Latents with shape (B 4 F H' W') using generator or not
    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, expand=False):
        if expand:
          shape = (batch_size, num_channels_latents, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
          latents = torch.randn(shape, device=device, dtype=dtype)
          latents = latents.expand(batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
      
        else:
          shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor) # (B 4 F H' W')
          if isinstance(generator, list) and len(generator) != batch_size:
              raise ValueError(
                  f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                  f" size of {batch_size}. Make sure the batch size matches the length of the generators."
              )
          if latents is None:
              rand_device = "cpu" if device.type == "mps" else device

              if isinstance(generator, list):
                  shape = shape
                  # shape = (1,) + shape[1:]
                  latents = [
                      torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                      for i in range(batch_size)
                  ]
                  latents = torch.cat(latents, dim=0).to(device)
              else:
                  latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
          else:
              if latents.shape != shape:
                  raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
              latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    @torch.no_grad()
    def cfg_plus_step(
        self, 
        model_output,
        uncond_model_output,
        timestep, 
        sample, 
        eta: float=0.0,  
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True, 
    ) -> DDIMSchedulerOutput:

        if self.scheduler.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = uncond_model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.scheduler.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self.scheduler._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                if device.type == "mps":
                    # randn does not work reproducibly on mps
                    variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
                    variance_noise = variance_noise.to(device)
                else:
                    variance_noise = torch.randn(
                        model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                    )
            variance = self.scheduler._get_variance(timestep, prev_timestep) ** (0.5) * eta * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]], # given
        video_length: Optional[int], # given
        height: Optional[int] = None, # given
        width: Optional[int] = None, # given
        num_inference_steps: int = 50, # given
        guidance_scale: float = 7.5, # given
        negative_prompt: Optional[Union[str, List[str]]] = None, # given
        num_videos_per_prompt: Optional[int] = 1, # 1
        eta: float = 0.0, # 0.0
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        # support controlnet
        controlnet_images: torch.FloatTensor = None, # given
        controlnet_image_index: list = [0], # given
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,   
        
        # support Image and cfg++
        is_video: bool = True,  
        cfg_plus: bool = False,
        cfg_plus_guidance_scale: float = 0.8,
        cfg_plus_temporal_guidance_scale: float = 0.8,
        savedir: str = None,
        
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct, 1. prompt is str or list, 2. height and width divisble by 8.
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        # (B 4 F H' W')
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
            expand = False,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        # eta, generator dictionary 
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # lamb_sds_start = 50.0
        # lamb_sds_end = 0.001
        
        # lamb_sds = lamb_schedule(lamb_sds_start, lamb_sds_end, num_inference_steps, device=device, type='log_linear')
        

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                down_block_additional_residuals = mid_block_additional_residual = None
                if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                    assert controlnet_images.dim() == 5

                    # controlnet latent와 prompt embedding이 들어감
                    controlnet_noisy_latents = latent_model_input # [2, 4, 16, 64, 64] (classifier_free_guidance)
                    controlnet_prompt_embeds = text_embeddings # [2, 77, 768] (uncond, cond)

                    controlnet_images = controlnet_images.to(latents.device) # [1, 3, N, 512, 512]

                    
                    controlnet_cond_shape    = list(controlnet_images.shape) # [1, 3, 16, 512, 512]
                    controlnet_cond_shape[2] = video_length
                    controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)
              
                    controlnet_conditioning_mask_shape    = list(controlnet_cond.shape) # [1, 1, 16, 512, 512]
                    controlnet_conditioning_mask_shape[1] = 1
                    controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)

                    assert controlnet_images.shape[2] >= len(controlnet_image_index)
                    controlnet_cond[:,:,controlnet_image_index] = controlnet_images[:,:,:len(controlnet_image_index)] # image for conditional image else 0
                    controlnet_conditioning_mask[:,:,controlnet_image_index] = 1 # 1 for conditional image else 0


                    print("1: ", controlnet_cond.shape, controlnet_conditioning_mask.shape)
                    
                    down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                        controlnet_noisy_latents, t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_mask=controlnet_conditioning_mask,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=False, return_dict=False,
                    )

                # predict the noise residual
                # down_block_additioanl_residuals, mid_block_additional_residual as controlNet
        
                # Predict the xt to x0 independently
                noise_pred = self.unet(
                    latent_model_input, t, 
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals = down_block_additional_residuals,
                    mid_block_additional_residual   = mid_block_additional_residual,
                    is_video = False,
                ).sample.to(dtype=latents_dtype)
                
                
                # perform guidance
                if do_classifier_free_guidance:
                    if cfg_plus:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + cfg_plus_guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if not cfg_plus:
                    # compute the previous noisy sample x_t -> x_t-1
                    output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                    latents = output.prev_sample
                    pred_x0 = output.pred_original_sample
                    with torch.no_grad():
                        pred_x0_video = self.decode_latents(pred_x0)
                        if output_type == "tensor":
                            pred_x0_video = torch.from_numpy(pred_x0_video)
                            # pred_x0_video = rearrange(pred_x0_video, 'b c f h w -> f c b h w')
                            if i == 0:
                                pred_total = pred_x0_video
                            else:
                                pred_total = torch.cat([pred_total, pred_x0_video], dim=0)
                    del output
                    
                else:
                    output = self.cfg_plus_step(noise_pred, noise_pred_uncond, t ,latents, **extra_step_kwargs)
                    # latents = output.prev_sample
                    pred_x0 = output.pred_original_sample
                    
                    del output
                    # with torch.no_grad():
                    #     pred_x0_video = self.decode_latents(pred_x0)
                    #     if output_type == "tensor":
                    #         pred_x0_video = torch.from_numpy(pred_x0_video)
                    #         # pred_x0_video = rearrange(pred_x0_video, 'b c f h w -> f c b h w')
                    #         if i == 0:
                    #             pred_total = pred_x0_video
                    #         else:
                    #             pred_total = torch.cat([pred_total, pred_x0_video], dim=0)
       
                                
                # Using pred_x0, guiding temporal guidance
                with torch.no_grad():
                    
                    t2 = t
                        
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    lamb_sds = (1 - alpha_prod_t) ** (0.5) / alpha_prod_t ** (0.5)
                        
                    for _ in range(1):
                        noise_add = torch.randn_like(pred_x0)
                        injection_timestep = t2
                        print(i, injection_timestep)
                        latents = self.scheduler.add_noise(pred_x0, noise_add, timesteps=injection_timestep)

                        noise_pred_t = self.unet(
                            latents, injection_timestep, 
                            encoder_hidden_states=text_embeddings[1:, :, :],
                            down_block_additional_residuals = down_block_additional_residuals,
                            mid_block_additional_residual   = mid_block_additional_residual,
                            is_video = True,
                        ).sample.to(dtype=latents_dtype)
                        

                        # print(lamb_sds)
                        pred_x0 = pred_x0 - 1.0 * lamb_sds * (noise_pred_t - noise_add) 
                        # output = self.scheduler.step(noise_pred_t, injection_timestep, latents, **extra_step_kwargs)
                        # pred_x0 = output.pred_original_sample
                    
                    
                    with torch.no_grad():
                        pred_x0_video = self.decode_latents(pred_x0)
                        if output_type == "tensor":
                            pred_x0_video = torch.from_numpy(pred_x0_video)
                            # pred_x0_video = rearrange(pred_x0_video, 'b c f h w -> f c b h w')
                            if i == 0:
                                pred_total = pred_x0_video
                            else:
                                pred_total = torch.cat([pred_total, pred_x0_video], dim=0)                    
                    
                    
                    prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                    alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                    pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * noise_pred_uncond
                    
                    print((1 - alpha_prod_t_prev) ** (0.5) / alpha_prod_t_prev ** (0.5))
   
                    latents = alpha_prod_t_prev ** (0.5) * pred_x0 + pred_sample_direction
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        os.makedirs(os.path.join(savedir, "pred_x0"), exist_ok=True)                
        save_videos_grid(pred_total, f"{savedir}/pred_x0/{prompt}.gif", n_rows=8)
        
        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
