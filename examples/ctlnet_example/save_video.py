import os
import time
import threading
from pathlib import Path
from typing import Optional, Union, Dict, List, Literal

import torch
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Queue, get_context

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderTiny
from transformers import CLIPVisionModelWithProjection

# Import your custom utility functions
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from my_image_utils import pil2tensor
from main_video import StreamDiffusionControlNetSample

from stream_info import *

# Global variables
inputs = []

def close_all_windows():
    """Close all windows and terminate any child processes."""
    print("Closing all windows...")
    cv2.destroyAllWindows()

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PyTorch Tensor to a PIL Image."""
    tensor = tensor.detach().cpu().float()
    tensor = (tensor * 255).clamp(0, 255).byte()
    tensor = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(tensor)

def screen(event: threading.Event(), height: int = 512, width: int = 512, video_file_path: str = None):
    """Capture frames from video and append to the inputs queue."""
    global inputs

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"Cannot open video file {video_file_path}")
        return

    try:
        while True:
            if event.is_set():
                print("Terminating video thread")
                break

            ret, frame = cap.read()
            if not ret:
                print("End of video file reached or frame capture failed.")
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_resized = img.resize((width, height))
            inputs.append(pil2tensor(img_resized))

            # Debugging: log frame capture and inputs length
            print(f"Frame captured: {ret}, inputs length: {len(inputs)}")

            time.sleep(0.01)  # Adjust frame interval
    finally:
        cap.release()
        close_all_windows()

def image_generation_process(
    queue: Queue,
    model_id_or_path: str,
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    seed: int,
    video_file_path: str,
    save_video: bool,
    t_index_list: List[int],
):
    """Process video frames and save generated images."""
    event = threading.Event()
    event.clear()

    # Initialize the ControlNet model and pipeline
    controlnet_pose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to("cuda")

    # IPAdapter's image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16).to("cuda")

    # Load the Stable Diffusion ControlNet pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id_or_path, controlnet=controlnet_pose, image_encoder=image_encoder).to("cuda", torch.float16)

    # Load LoRA and configure pipeline
    pipe.load_ip_adapter('h94/IP-Adapter', subfolder="models", weight_name="ip-adapter_sd15.bin", torch_dtype=torch.float16)
    pipe.set_ip_adapter_scale(0.8)

    pipe.load_lora_weights(LORA_PATH, adapter_name=LORA_NAME)
    pipe.set_adapters([LORA_NAME], adapter_weights=[1.0])

    stream = StreamDiffusionControlNetSample(
        pipe, t_index_list=t_index_list, torch_dtype=torch.float16,
        width=width, height=height, acceleration=acceleration, model_id_or_path=model_id_or_path
    )

    # Prepare the pipeline for inference
    stream.prepare(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50,
        guidance_scale=1.2, delta=1.0
    )
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    # Start the video reading thread
    video_thread = threading.Thread(target=screen, args=(event, height, width, video_file_path))
    video_thread.start()

    frame_count = 0
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = f"processed_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    while True:
        try:
            if event.is_set():
                break

            if len(inputs) < frame_buffer_size:
                time.sleep(0.01)
                continue

            # Debugging: log inputs status
            print(f"Processing inputs: {len(inputs)} frames available for processing.")

            input_batch = torch.cat(inputs[-frame_buffer_size:])
            inputs.clear()

            try:
                output_images = stream.ctlimg2img(ctlnet_image=input_batch)
            except Exception as e:
                print(f"Error in ctlimg2img: {e}")
                break

            output_images = [output_images] if frame_buffer_size == 1 else output_images

            for output_image in output_images:
                output_image_pil = tensor_to_pil(output_image)
                output_image_pil.save(os.path.join(output_folder, f"frame_{frame_count:05d}.png"))
                frame_count += 1

            time.sleep(0.01)  # Adjust processing speed as needed

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
        finally:
            event.set()
            video_thread.join()
            close_all_windows()
            break

def main(
    model_id_or_path: str = MODEL_PATH,
    lora_dict: Optional[Dict[str, float]] = {LORA_PATH: 1.0},
    prompt: str = PROMPT,
    negative_prompt: str = "low quality, blurry",
    frame_buffer_size: int = 1,
    width: int = SD_WIDTH,
    height: int = SD_HEIGHT,
    acceleration: Literal["none", "xformers", "tensorrt"] = "none",
    seed: int = 2,
    video_file_path: str = VIDEO_PATH,
    save_video: bool = True,
    t_index_list: List[int] = T_INDEXT_LIST,  # Example values
) -> None:
    """Main function for running video processing."""
    ctx = get_context('spawn')
    queue = ctx.Queue()

    image_process = ctx.Process(
        target=image_generation_process,
        args=(
            queue, model_id_or_path, prompt, negative_prompt,
            frame_buffer_size, width, height, acceleration, seed,
            video_file_path, save_video, t_index_list
        ),
    )
    image_process.start()
    image_process.join()

if __name__ == "__main__":
    main()
