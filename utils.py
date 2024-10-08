import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist

from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from convert_lora_safetensor_to_diffusers import convert_lora, load_diffusers_lora, load_diffusers_lora_unet

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def load_weights(
    unet,
    vae,
    text_encoder,
    # motion module
    motion_module_path         = "",
    motion_module_lora_configs = [],
    # domain adapter
    adapter_lora_path          = "",
    adapter_lora_scale         = 1.0,
    # image layers
    dreambooth_model_path      = "",
    lora_model_path            = "",
    lora_alpha                 = 0.8,
):
    # motion module
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        if motion_module_path.endswith(".safetensors"):
          motion_module_state_dict = {}
          with safe_open(motion_module_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    motion_module_state_dict[key] = f.get_tensor(key)
        else:
          motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
          
        motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        motion_module_state_dict = {k.replace("module.", ""):v for k, v in motion_module_state_dict.items()}
        unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
        unet_state_dict.pop("animatediff_config", "")
    
    missing, unexpected = unet.load_state_dict(unet_state_dict, strict=False)
    print("missing: ", len(missing), " unexpected: ", len(unexpected))
    assert len(unexpected) == 0
    del unet_state_dict

    # base model
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
            
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, vae.config)
        vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, unet.config)
        unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict
        
    # # lora layers
    # if lora_model_path != "":
    #     print(f"load lora model from {lora_model_path}")
    #     assert lora_model_path.endswith(".safetensors")
    #     lora_state_dict = {}
    #     with safe_open(lora_model_path, framework="pt", device="cpu") as f:
    #         for key in f.keys():
    #             lora_state_dict[key] = f.get_tensor(key)
                
    #     # convert lora function은 각 layer에 맞춰서 weight를 더해주는 작업을 함
    #     animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
    #     del lora_state_dict

    # # domain adapter lora
    # if adapter_lora_path != "":
    #     print(f"load domain lora from {adapter_lora_path}")
        
    #     if adapter_lora_path.endswith(".safetensors"):
    #         domain_lora_state_dict = {}
    #         with safe_open(adapter_lora_path, framework="pt", device="cpu") as f:
    #             for key in f.keys():
    #                 domain_lora_state_dict[key] = f.get_tensor(key)
    #     else:  
    #       domain_lora_state_dict = torch.load(adapter_lora_path, map_location="cpu")
          
    #     domain_lora_state_dict = domain_lora_state_dict["state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
    #     domain_lora_state_dict.pop("animatediff_config", "")

    #     animation_pipeline = load_diffusers_lora(animation_pipeline, domain_lora_state_dict, alpha=adapter_lora_scale)

    # # motion module lora
    # for motion_module_lora_config in motion_module_lora_configs:
    #     path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
    #     print(f"load motion LoRA from {path}")
    #     motion_lora_state_dict = torch.load(path, map_location="cpu")
    #     motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
    #     motion_lora_state_dict.pop("animatediff_config", "")

    #     animation_pipeline = load_diffusers_lora(animation_pipeline, motion_lora_state_dict, alpha)

    return unet