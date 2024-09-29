#pipeline_td cn
import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

import os
import sys
from pathlib import Path
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from my_image_utils import pil2tensor

from transformers import CLIPVisionModelWithProjection


from streamdiffusion.image_filter import SimilarImageFilter

class StreamDiffusion:
    def __init__(
        self,
        pipe: DiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        use_controlnet: bool = False,

    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        # print("scheduler")
        # print(self.scheduler)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.inference_time_ema = 0

        self.sdxl = type(self.pipe) is StableDiffusionXLPipeline

        self.use_controlnet = use_controlnet
        if self.use_controlnet:
            self.controlnet = pipe.controlnet
        else:
            self.controlnet = None
        self.controlnet_conditioning_scale = 1.0
        self.controlnet_start_step: int = 0
        self.controlnet_end_step: int = 49


    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            self.control_image_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    3,
                    self.height,
                    self.width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None
            self.control_image_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.sdxl:
            self.add_text_embeds = encoder_output[2]
            original_size = (self.height, self.width)
            crops_coords_top_left = (0, 0)
            target_size = (self.height, self.width)
            text_encoder_projection_dim = int(self.add_text_embeds.shape[-1])
            self.add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=encoder_output[0].dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        added_cond_kwargs,
        idx: Optional[int] = None,
        control_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent


        if self.use_controlnet and self.controlnet is not None:
            down_samples, mid_sample = self.controlnet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                controlnet_cond=control_image,
                guess_mode=False,
                return_dict=False,
                **added_cond_kwargs,  # Pass added_cond_kwargs to controlnet

            )
            down_block_res_samples = [
                down_sample * self.controlnet_conditioning_scale
                for down_sample in down_samples
            ]
            mid_block_res_sample = self.controlnet_conditioning_scale * mid_sample
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        else:
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        output_latent = self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return output_latent


    def predict_x0_batch(self, x_t_latent: torch.Tensor, control_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        # print("Entering predict_x0_batch")
        added_cond_kwargs = {}
        prev_latent_batch = self.x_t_latent_buffer

        if self.use_denoising_batch:
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                # print("Updated x_t_latent with previous batch")

            if self.sdxl:
                added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), "time_ids": self.add_time_ids.to(self.device)}
                # print("Added condition kwargs for SDXL")

            if self.use_controlnet:
                # print(f"Initial Control Image Buffer shape: {self.control_image_buffer.shape if self.control_image_buffer is not None else 'None'}")
                if control_image is not None:
                    control_image = control_image.to(self.device, dtype=self.dtype)
                    if control_image.dim() == 3:
                        control_image = control_image.unsqueeze(0)
                    control_image = torch.cat((control_image, self.control_image_buffer), dim=0)
                    # print(f"Concatenated current control image with buffer. New shape: {control_image.shape}")
                else:
                    control_image = self.control_image_buffer
                    # print("Using control image buffer as control image")

            t_list = self.sub_timesteps_tensor
            x_t_latent = x_t_latent.to(self.device)
            t_list = t_list.to(self.device)
            # print("Device transfer complete for tensors")

            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs, control_image=control_image)

            #UPDATE BUFFER
            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
                #new new
                if self.use_controlnet:
                    self.control_image_buffer = control_image[:-self.frame_bff_size]

            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
                self.control_image_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                if self.sdxl:
                    added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), "time_ids": self.add_time_ids.to(self.device)}
                # x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs)
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs, control_image=control_image)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred
        return x_0_pred_out

    @torch.no_grad()
    def __call__(
        self,
        x: Union[torch.Tensor, Image.Image, np.ndarray] = None,
        control_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if x is not None:
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
            x_t_latent = self.encode_image(x)
        else:
            # TODO: check the dimension of x_t_latent
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        x_0_pred_out = self.predict_x0_batch(x_t_latent, control_image)
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output
        if self.device == "cuda":
            end.record()
            torch.cuda.synchronize()
            inference_time = start.elapsed_time(end) / 1000
            self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1, control_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            ),
            control_image,
        )
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output


    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)


class StreamDiffusionControlNetSample(StreamDiffusion):
    def __init__(self,
                 pipe: StableDiffusionPipeline,
                 t_index_list: List[int],
                 torch_dtype: torch.dtype = torch.float16,
                 width: int = 512,
                 height: int = 512,
                 do_add_noise: bool = True,
                 use_denoising_batch: bool = True,
                 frame_buffer_size: int = 1,
                 cfg_type: Literal["none", "full", "self", "initialize"] = "self",
                 acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
                 engine_dir: Optional[Union[str, Path]] = "engines",
                 model_id_or_path: str = "Lykon/dreamshaper-8-lcm",
                 ip_adapter=None):
        super().__init__(pipe,
                         t_index_list,
                         torch_dtype,
                         width,
                         height,
                         do_add_noise,
                         use_denoising_batch,
                         frame_buffer_size,
                         cfg_type,
                         )
        self.ip_adapter = ip_adapter
        if pipe.controlnet != None:
            self.controlnet = pipe.controlnet
        self.input_latent = None
        self.ctl_image_t_buffer = None
        self.added_cond_kwargs = None
        self.acceleration = acceleration
        self.engine_dir = engine_dir
        self.model_id_or_path = model_id_or_path

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
        ip_adapter_image=None,
        target_image_weight: float = 0.8,
        initial_steps_ratio: float = 0.3,
    ) -> None:
        self.do_classifier_free_guidance = False
        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta
        self.do_classifier_free_guidance = self.is_do_classifer_free_guicance()
        self.target_image_weight = target_image_weight
        self.initial_steps_ratio = initial_steps_ratio

        # IPAdapterのため
        if self.ip_adapter:
            ip_adapter_image = ip_adapter_image.resize((self.height, self.width))

            # SD IPADAPTERIMPL
            num_images_per_prompt = 1

            if ip_adapter_image is not None:
                image_embeds, negative_image_embeds = self.pipe.encode_image(ip_adapter_image, "cuda", num_images_per_prompt)

            print("image_embeded:{}".format(image_embeds.shape))

        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = torch.zeros(
                (self.frame_bff_size, 4, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
            #self.x_t_latent_buffer = None

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

        if self.ip_adapter:
            # IPADAPTER ORIGINAL IMPL
            image_embeds = image_embeds.repeat(self.batch_size, 1, 1)
            if self.do_classifier_free_guidance:
                negative_image_embeds = negative_image_embeds.repeat(self.batch_size, 1, 1)

                image_embeds = torch.cat([negative_image_embeds, image_embeds])

            # SD IPADAPTER IMPL
            self.added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )



        ### TODO: 高速化
        try:
            if self.acceleration == "xformers":
                self.pipe.enable_xformers_memory_efficient_attention()
            if self.acceleration == "tensorrt":
                from polygraphy import cuda
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(self.model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}"
                    else:
                        return f"{self.model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}"

                engine_dir = Path(self.engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=self.model_id_or_path,
                        max_batch_size=self.trt_unet_batch_size,
                        min_batch_size=self.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=self.model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else self.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else self.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=self.model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else self.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else self.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=self.device,
                        max_batch_size=self.trt_unet_batch_size,
                        min_batch_size=self.trt_unet_batch_size,
                        embedding_dim=self.text_encoder.config.hidden_size,
                        unet_dim=self.unet.config.in_channels,
                    )
                    compile_unet(
                        self.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=self.trt_unet_batch_size,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=stream.trt_unet_batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=stream.trt_unet_batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )

                cuda_steram = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_steram, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_steram,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
                if self.acceleration == "sfast":
                    from streamdiffusion.acceleration.sfast import (
                        accelerate_with_stable_fast,
                    )

                    stream = accelerate_with_stable_fast(stream)
                    print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")


    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
        image=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        latent_model_input = x_t_latent_plus_uc
        controlnet_prompt_embeds = self.prompt_embeds
        control_model_input = latent_model_input
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            t_list,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=1,
            guess_mode=False,
            return_dict=False,
        )

        model_pred = self.unet(
            x_t_latent_plus_uc,
            t_list,
            encoder_hidden_states=self.prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
            added_cond_kwargs=self.added_cond_kwargs
        )[0]

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def is_do_classifer_free_guicance(self):
        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True
        return do_classifier_free_guidance

    @torch.no_grad()
    def update_prompt(self, prompt: str, negative_prompt) -> None:
        do_classifier_free_guidance = self.is_do_classifer_free_guicance()

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

    def predict_x0_batch(self, x_t_latent: torch.Tensor,
                         image=None) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer
        # todo とりあえず埋める。
        if self.ctl_image_t_buffer is None or self.x_t_latent_buffer.shape[0] >= self.ctl_image_t_buffer.shape[0]:
            self.ctl_image_t_buffer = image.repeat(self.x_t_latent_buffer.shape[0], 1, 1, 1)

        prev_ctl_image_t_buffer = self.ctl_image_t_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )

                images = torch.cat(
                    (image, prev_ctl_image_t_buffer), dim=0
                )

            if images.shape[2:] != x_t_latent.shape[2:]:
                images = torch.nn.functional.interpolate(images, size=x_t_latent.shape[2:])

            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list, image=images)

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
                if self.cfg_type == "full":
                    self.ctl_image_t_buffer = images[:-2]
                else:
                    self.ctl_image_t_buffer = images[:-1]
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx, image=image)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    @torch.no_grad()
    def ctlimg2img(self, batch_size: int = 1, ctlnet_image=None, keep_latent=False) -> torch.Tensor:
        if not keep_latent:
            self.input_latent = torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        else:
            if self.input_latent is None:
                latent = torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                    device=self.device, dtype=self.dtype
                )
                self.input_latent = latent

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tstart = time.time()

        # コントロールネット用の計算
        num_images_per_prompt = 1
        batch_size = 1
        guess_mode = False
        if self.pipe.controlnet != None:
            timage = self.pipe.prepare_image(
                image=ctlnet_image,
                width=self.width,
                height=self.height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=self.device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

        ctlnet_image = timage

        if self.ip_adapter and self.added_cond_kwargs and 'image_embeds' in self.added_cond_kwargs:
            image_embeds = self.added_cond_kwargs['image_embeds']
            image_embeds = image_embeds * self.target_image_weight
            self.added_cond_kwargs['image_embeds'] = image_embeds
        x_0_pred_out = self.predict_x0_batch(self.input_latent, ctlnet_image)

        tstart = time.time()
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        tstart = time.time()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output
