import gc
import os
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        sdxl: bool = None
    ):
        self.sd_turbo = "turbo" in model_id_or_path
        self.sdxl = sdxl if sdxl is not None else "xl" in model_id_or_path
        self.default_tiny_vae = "madebyollin/taesdxl" if self.sdxl else "madebyollin/taesd"

        if mode == "txt2img" and cfg_type != "none":
            raise ValueError(f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}")
        if mode == "img2img" and not use_denoising_batch:
            raise NotImplementedError("img2img mode must use denoising batch for now.")

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size if use_denoising_batch else frame_buffer_size
        )
        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        if hasattr(self.stream.unet, 'config'):
            self.stream.unet.config.addition_embed_type = None

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(self.stream.unet, device_ids=device_ids)

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, (str, Image.Image)):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        images = postprocess_image(image_tensor.cpu(), output_type=output_type)
        return images if self.frame_buffer_size > 1 else images[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ) -> StreamDiffusion:
        if self.sdxl:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(model_id_or_path)
            except ValueError:
                pipe = StableDiffusionXLPipeline.from_single_file(model_id_or_path)
            pipe = pipe.to(device=self.device, dtype=self.dtype)
        else:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path)
            except ValueError:
                pipe = StableDiffusionPipeline.from_single_file(model_id_or_path)
            pipe = pipe.to(device=self.device, dtype=self.dtype)

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )

        if not self.sd_turbo:
            if use_lcm_lora:
                stream.load_lcm_lora(pretrained_model_name_or_path_or_dict=lcm_lora_id)
                stream.fuse_lora()

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)
                    print(f"Use LoRA: {lora_name} in weights {lora_scale}")

        if use_tiny_vae:
            vae_model = vae_id if vae_id is not None else self.default_tiny_vae
            stream.vae = AutoencoderTiny.from_pretrained(vae_model).to(
                device=pipe.device, dtype=pipe.dtype
            )

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                pass  # TensorRT accelerationコードを省略
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast
                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0:
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
