import os
import sys
import time
import threading
import json
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue, get_context
from multiprocessing.connection import Connection
import torch
import PIL.Image
import mss
import fire
import tkinter as tk
import cv2
import traceback
from screeninfo import get_monitors
from datetime import datetime
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

from diffusers import AutoencoderTiny, StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from my_image_utils import pil2tensor
from transformers import CLIPVisionModelWithProjection
from PIL import Image

from main_video import StreamDiffusionControlNetSample, close_all_windows, monitor_setting_process, apply_gamma_correction, normalize_image

inputs = []

def read_video(
    event: threading.Event(),
    video_file_path: str,
    height: int = 512,
    width: int = 512,
    upper_fps: int = 1,
    monitor_info: Dict[str, Any] = None,
    resize_mode: bool = True,
    close_queue: Queue = None,  # 追加
):
    global inputs

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"Cannot open video file {video_file_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")


    cv2.namedWindow("Video Input", cv2.WINDOW_NORMAL)
    if monitor_info:
        monitor_x = monitor_info['left']
        monitor_y = monitor_info['top']
        cv2.moveWindow("Video Input", monitor_x, monitor_y)
        cv2.resizeWindow("Video Input", width, height)

    try:
        while True:
            if event.is_set():
                print("terminate video thread")
                break

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("End of video file reached.")
                if close_queue:  # 終了を通知
                    close_queue.put(True)
                break
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
                # continue

            img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if resize_mode:
                img_resized = img.resize((width, height))
            else:
                img_width, img_height = img.size
                left_crop = (img_width - width) // 2
                top_crop = (img_height - height) // 2
                right_crop = left_crop + width
                bottom_crop = top_crop + height
                img_cropped = img.crop((left_crop, top_crop, right_crop, bottom_crop))
                img_resized = img_cropped

            cv2.imshow("Video Input", cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == ord('q'):
                break

            inputs.append(pil2tensor(img_resized))

            interval = time.time() - start_time
            fps_interval = 1.0 / upper_fps
            if interval < fps_interval:
                sleep_time = fps_interval - interval
                time.sleep(sleep_time)
    finally:
        cap.release()
        cv2.destroyWindow("Video Input")
        print('exit : read_video')

def load_config_from_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, 'r') as file:
        return json.load(file)

# 出力ディレクトリを作成する関数
def create_output_dir(video_file_path: str, save_dir: str) -> str:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]
    output_dir = os.path.join(save_dir, base_name, current_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# JSONをコピーする関数
def copy_config_to_output(json_path: str, output_dir: str):
    shutil.copy(json_path, os.path.join(output_dir, os.path.basename(json_path)))

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    config: Dict[str, Any],  # 設定をJSONから受け取る
    prompt_queue,
    monitor_receiver: Connection,
) -> None:
    global inputs
    global base_img

    instep = config["INSTEP"]
    image_index = 0
    output_dir = create_output_dir(config["VIDEO_PATH"], config["SAVE_PNG_DIR"])
    print(f"Output directory created: {output_dir}")

    t_index_list = config["T_INDEX_LIST"]
    guidance_scale = config["GUIDANCE_SCALE"]
    delta = config["DELTA"]
    prompt = config["PROMPT"]
    negative_prompt = config["NEGATIVE_PROMPT"]

    adapter = True
    ip_adapter_image_filepath = config["IP_ADAPTER_IMAGE"]

    # モデル準備
    controlnet_pose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16
    ).to("cuda")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config["MODEL_PATH"],
        controlnet=controlnet_pose,
        image_encoder=image_encoder
    ).to(device=torch.device("cuda"), dtype=torch.float16)

    pipe.load_lora_weights(config["LORA_PATH"], adapter_name=config["LORA_NAME"])
    pipe.set_adapters([config["LORA_NAME"]], adapter_weights=[1.0])

    stream = StreamDiffusionControlNetSample(
        pipe,
        t_index_list=t_index_list,
        torch_dtype=torch.float16,
        cfg_type="none",
        width=config["SD_WIDTH"],
        height=config["SD_HEIGHT"],
        ip_adapter=adapter,
        acceleration="none",
        model_id_or_path=config["MODEL_PATH"],
    )

    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    ip_adapter_image = load_image(ip_adapter_image_filepath)

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=instep,
        guidance_scale=guidance_scale,
        delta=delta,
        ip_adapter_image=ip_adapter_image
    )

    event = threading.Event()
    event.clear()

    # 入力ソースに応じて処理を分岐
    input_source = 'video'
    monitor_list = monitor_receiver.recv()
    monitor_info = monitor_list[0]

    input_thread = threading.Thread(target=read_video, args=(event, config["VIDEO_PATH"], config["SD_HEIGHT"], config["SD_WIDTH"], config["UPPER_FPS"], monitor_info))
    input_thread.start()

    total_frames = 0
    while True:
        try:
            if not close_queue.empty():
                print("Termination signal received.")
                break
            if len(inputs) < config["FRAME_BUFFER_SIZE"]:
                time.sleep(config["FPS_INTERVAL"])
                continue

            start_time = time.time()
            sampled_inputs = [inputs.pop() for _ in range(config["FRAME_BUFFER_SIZE"])]
            input_batch = torch.cat(sampled_inputs)
            input = input_batch.to(device=stream.device, dtype=stream.dtype)
            inputs.clear()

            output_images = stream.ctlimg2img(ctlnet_image=input)
            total_frames += len(output_images)
            print(f"\rtotal {total_frames}", end='', flush=True)

            for output_image in output_images:
                queue.put(output_image, block=False)
                if isinstance(output_image, torch.Tensor):
                    output_image_np = normalize_image(output_image.squeeze().cpu().numpy())
                    output_image_np = np.clip(output_image_np * 255, 0, 255).astype(np.uint8)
                    output_image_np = np.moveaxis(output_image_np, 0, -1)
                    output_pil_image = Image.fromarray(output_image_np)
                    image_index += 1
                    output_pil_image.save(os.path.join(output_dir, f"output_image_{image_index:04d}.png"))

            process_time = time.time() - start_time
            if process_time <= config["FPS_INTERVAL"]:
                time.sleep(config["FPS_INTERVAL"] - process_time)
            fps_queue.put(1 / process_time)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            traceback.print_exc()
            break

def main(json_config: str):
    config = load_config_from_json(json_config)
    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    prompt_queue = ctx.Queue()
    close_queue = Queue()
    monitor_sender, monitor_receiver = ctx.Pipe()

    process1 = ctx.Process(
        target=image_generation_process,
        args=(queue, fps_queue, close_queue, config, prompt_queue, monitor_receiver)
    )
    process1.start()

    monitor_process = ctx.Process(target=monitor_setting_process, args=(config["SD_WIDTH"], config["SD_HEIGHT"], monitor_sender))
    monitor_process.start()
    monitor_process.join()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    process2.join()
    close_queue.put(True)
    process1.join(5)
    if process1.is_alive():
        process1.terminate()
    process1.join()

if __name__ == "__main__":
    fire.Fire(main)
