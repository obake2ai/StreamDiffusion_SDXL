#wrapper td  cn 
import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
# from diffusers import TCDScheduler
from diffusers import DDIMScheduler

from PIL import Image
import time
import re
from pipeline_td import StreamDiffusion

# from streamdiffusion import StreamDiffusion
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
        device: Literal["cpu", "mps", "cuda"] = "cuda",
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
        sdxl: bool = None,
        scheduler_name: str = "EulerAncestral",  # Default scheduler name
        use_karras_sigmas: bool = False,  # Default setting for Karras sigmas
        use_controlnet: bool = False,
        controlnet_model: Optional[str] = None,
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        """
        # self.sd_turbo = "turbo" in model_id_or_path
        self.sd_turbo = "turbo" in model_id_or_path.lower() or "sdxs" in model_id_or_path.lower() or "lightning" in model_id_or_path.lower()
        
        if sdxl is None:
            self.sdxl = "xl" in model_id_or_path.lower()
        else:
            self.sdxl = sdxl
        
        self.vae_model_id = "madebyollin/taesdxl" if self.sdxl else "madebyollin/taesd"
        # print(f"VAE Model ID: {self.vae_model_id}")
        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "img2img mode must use denoising batch for now."
                )

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
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
            scheduler_name=scheduler_name,
            use_karras_sigmas=use_karras_sigmas,
            use_controlnet=use_controlnet,
            controlnet_model=controlnet_model
        )

        if hasattr(self.stream.unet, 'config'):
            self.stream.unet.config.addition_embed_type = None

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
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
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
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
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
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
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
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
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]

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
        scheduler_name: str = "EulerAncestral",  # Default scheduler name
        use_karras_sigmas: bool = False,  # Default setting for Karras sigmas
        use_controlnet: bool = False,  # Default setting for ControlNet
        controlnet_model: Optional[str] = None,

    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.

        Returns
        -------
        StreamDiffusion
            The loaded model.
        """
        print("\n====================================")
        print("\033[36m...Loading models...\033[0m")
        model_name = os.path.basename(model_id_or_path)

        use_sdxl = self.sdxl  # This should be an attribute of the class or set elsewhere
        self.use_controlnet = use_controlnet

        if use_controlnet:
            # Determine the specific ControlNet model to load
            controlnet_id = controlnet_model if controlnet_model else "lllyasviel/sd-controlnet-canny"
            cn_model_name = os.path.basename(controlnet_id)
            print(f"\n\033[36m...Loading ControlNet model: {cn_model_name}\033[0m") 
            try:
                # Try loading the ControlNet model using from_pretrained
                controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to(device=self.device, dtype=self.dtype)
                print(f"Successfully loaded ControlNet model from Huggingface")
                print(f"\033[92m{controlnet_id}\033[0m\n")
            except Exception as e:
                pretrained_error = e
                # print(f"Failed to load ControlNet model {controlnet_id} using from_pretrained due to: {e}")
                # Attempt to load from a single file if from_pretrained fails
                try:
                    # print(f"\nLoading ControlNet model from single file: {controlnet_id}")
                    controlnet = ControlNetModel.from_single_file(controlnet_id).to(device=self.device, dtype=self.dtype)
                    print(f"Successfully loaded ControlNet model from local directory:")
                    print(f"\033[92m{controlnet_id}\033[0m\n")
                except Exception as e:
                    print(f"Failed to load ControlNet model {controlnet_id} using from Huggingface (from_pretrained) due to: {pretrained_error}")
                    print(f"Failed to load ControlNet model from local directory (from_single_file) {controlnet_id} due to: {e}")
                    traceback.print_exc()
                    exit()
            

            # Decide on which pipeline to use based on use_sdxl flag
            if use_sdxl:
                print(f"\n\033[36m...Loading SDXL StableDiffusion model with ControlNet: {model_name}\033[0m")
                pipeline_class = StableDiffusionXLControlNetPipeline
            else:
                print(f"\n\033[36m...Loading StableDiffusion model with ControlNet: {model_name}\033[0m")
                pipeline_class = StableDiffusionControlNetImg2ImgPipeline

            # Load the appropriate pipeline
            try:
                pipe = pipeline_class.from_pretrained(
                    model_id_or_path,
                    controlnet=controlnet,
                    # torch_dtype=torch.float16
                ).to(device=self.device, dtype=self.dtype)
                print(f"Successfully loaded {model_name} from Huggingface:")
                print(f"\033[92m{model_id_or_path}\033[0m\n")
            except ValueError:  # Fallback to loading from a single file if from_pretrained fails
                pipe = pipeline_class.from_single_file(
                    model_id_or_path,
                    controlnet=controlnet,
                    # torch_dtype=torch.float16
                ).to(device=self.device, dtype=self.dtype)
                print(f"Successfully loaded {model_name} from local directory:")
                print(f"\033[92m{model_id_or_path}\033[0m\n")
            except Exception as e:
                print(f"Failed to load model with ControlNet due to: {e}")
                traceback.print_exc()
                exit()

        else:
            pipeline_class = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline
            model_type = "SDXL" if use_sdxl else "SD"

            print(f"\n\033[36m...Loading {model_type} model: {model_name}\033[0m")
            try:
                pipe = pipeline_class.from_pretrained(model_id_or_path)
                print(f"Successfully loaded {model_name} from Hugging Face:")
                print(f"\033[92m{model_id_or_path}\033[0m\n")
            except ValueError:
                try:
                    pipe = pipeline_class.from_single_file(model_id_or_path)
                    print(f"Successfully loaded {model_name} from local directory:")
                    print(f"\033[92m{model_id_or_path}\033[0m\n")
                except Exception as e:
                    print(f"Failed to load {model_type} model from both local and Hugging Face due to: {e}")
                    traceback.print_exc()
                    exit()
            finally:
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
            use_controlnet=use_controlnet,
        )
        if not self.sd_turbo:
            if use_lcm_lora:
                print(f"\n\033[36m...Loading LCM-LoRA Model: {os.path.basename(lcm_lora_id) if lcm_lora_id else 'latent-consistency/lcm-lora-sdv1-5'}\033[0m")
                try:
                    if lcm_lora_id is not None:
                        stream.load_lcm_lora(
                            pretrained_model_name_or_path_or_dict=lcm_lora_id
                        )
                    else:
                        stream.load_lcm_lora()
                    stream.fuse_lora()
                    print(f"Successfully loaded LCM-LoRA model:")
                    print(f"\033[92m{lcm_lora_id}\033[0m\n")
                except Exception as e:
                    print(f"\nERROR loading Local LCM-LoRA: {e}\n")
        if lora_dict is not None:
            try:
                
                    for lora_name, lora_scale in lora_dict.items():
                        # print("\n====================================")
                        print(f"\n\033[36m...Loading additional LoRA Model:  {os.path.basename(lora_name)}, Weight: {lora_scale}\033[0m")
                        stream.load_lora(lora_name)
                        stream.fuse_lora(lora_scale=lora_scale)
                        print(f"Successfully loaded LoRA model:")
                        print(f"\033[92m{lora_name}\033[0m\n")
            except Exception as e:
                print(f"\nERROR loading LoRA Models: {e}\n")
                pass
            # print("====================================\n")

        if hasattr(stream.unet, 'config'):
            stream.unet.config.addition_embed_type = None


        if use_tiny_vae:
            print(f"\n\033[36m...Loading VAE: {vae_id if vae_id else self.vae_model_id}\033[0m")
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                stream.vae = AutoencoderTiny.from_pretrained(self.vae_model_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            print(f"Successfully loaded VAE model:")
            print(f"\033[92m{vae_id if vae_id else self.vae_model_id}\033[0m\n")
        # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        # print("scheduler")
        # print(pipe.scheduler)
        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                #bl
                print("\n====================================")
                print("Using TensorRT...")
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
                    width: int,
                    height: int,
                ):
                    if width == 512 and height == 512:
                        resolution = ""
                    else:
                        resolution = f"--width-{width}--height-{height}"
                    maybe_path = Path(model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}{resolution}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}{resolution}"


                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "vae_decoder.engine",
                )
                # print("!!! \033[1mSTARTING TENSORRT\033[0m !!!\n--------------------------------")
                # print(f"self.sdxl value: {self.sdxl}")
                engine_build_options = {
                    "opt_image_height": self.height,
                    "opt_image_width": self.width,
                }

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                        # is_xl=self.sdxl,

                    )
                    print("\nCompiling TensorRT UNet...\nThis may take a moment...\n")
                    time.sleep(1)
                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                        engine_build_options=engine_build_options,
                        # is_xl=self.sdxl,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    print("\nCompiling TensorRT VAE Decoder...\n")
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        engine_build_options=engine_build_options,

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
                    print("\nCompiling TensorRT VAE Encoder...\n")
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        engine_build_options=engine_build_options,
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
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0: # Random seed
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



    def load_scheduler(self, scheduler_name: str, use_karras_sigmas: bool = False):
        """
        Dynamically loads a scheduler based on the given name, with the ability to handle
        various initialization parameters such as Karras sigmas.

        Parameters:
        - scheduler_name (str): The name of the scheduler to load.
        - use_karras_sigmas (bool): Whether to use Karras sigmas for applicable schedulers.

        Returns:
        - A scheduler instance from the diffusers library.
        """
        try:
            # Mapping scheduler names to their corresponding module paths and initialization parameters
            scheduler_map = {
                "LMS": ("diffusers.LMSDiscreteScheduler", {"use_karras_sigmas": use_karras_sigmas}),
                "DPMSolverMultistep": ("diffusers.DPMSolverMultistepScheduler", {"use_karras_sigmas": use_karras_sigmas}),
                "KDPM2": ("diffusers.KDPM2DiscreteScheduler", {"use_karras_sigmas": use_karras_sigmas}),
                "KDPM2Ancestral": ("diffusers.KDPM2AncestralDiscreteScheduler", {"use_karras_sigmas": use_karras_sigmas}),
                "Euler": ("diffusers.EulerDiscreteScheduler", {}),
                "EulerAncestral": ("diffusers.EulerAncestralDiscreteScheduler", {}),
                "Heun": ("diffusers.HeunDiscreteScheduler", {}),
                "DEISMultistep": ("diffusers.DEISMultistepScheduler", {}),
                "UniPCMultistep": ("diffusers.UniPCMultistepScheduler", {})
            }

            if scheduler_name not in scheduler_map:
                # Fallback to default scheduler if specified one is not found
                from diffusers import EulerDiscreteScheduler
                print(f"Scheduler '{scheduler_name}' not found. Fallback to default 'EulerDiscreteScheduler'.")
                return EulerDiscreteScheduler()

            # Dynamic import based on the scheduler_map
            module_name, class_name = scheduler_map[scheduler_name][0].rsplit('.', 1)
            scheduler_module = __import__(module_name, fromlist=[class_name])
            scheduler_class = getattr(scheduler_module, class_name)
            
            # Initialize the scheduler with the specified parameters
            scheduler = scheduler_class(**scheduler_map[scheduler_name][1])
            print(f"Loaded scheduler: {scheduler_name} with params: {scheduler_map[scheduler_name][1]}")
            return scheduler

        except Exception as e:
            # Handle unexpected errors in scheduler loading
            from diffusers import EulerDiscreteScheduler
            print(f"Error loading scheduler '{scheduler_name}': {str(e)}. Using default 'EulerDiscreteScheduler'.")
            return EulerDiscreteScheduler()
