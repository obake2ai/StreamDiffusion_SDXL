import os
import sys
import time
import threading
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

from main_video import StreamDiffusionControlNetSample, close_all_windows

###############################################
# プロンプトはここ
###############################################
box_prompt = "xshingogirl"
###############################################

UPEER_FPS = 2
fps_interval = 1.0 / UPEER_FPS
inputs = []
top = 0
left = 0


def screen(
    event: threading.Event(),
    height: int = 512,
    width: int = 512,
    monitor: Dict[str, int] = {"top": 300, "left": 200, "width": 512 * 2, "height": 512 * 2},
):
    global inputs

    with mss.mss() as sct:
        while True:
            if event.is_set():
                print("terminate read thread")
                break
            start_time = time.time()
            img = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            img = img.resize((height, width))
            inputs.append(pil2tensor(img))
            interval = time.time() - start_time
            fps_interval = 1.0 / UPEER_FPS
            if interval < fps_interval:
                sleep_time = fps_interval - interval
                time.sleep(sleep_time)

    print('exit : screen')


def camera(
    event: threading.Event(),
    height: int = 512,
    width: int = 512,
    monitor_info: Dict[str, Any] = None,
):
    global inputs

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # カメラウィンドウを設定
    cv2.namedWindow("Camera Input", cv2.WINDOW_NORMAL)
    if monitor_info:
        # ウィンドウを別モニターに移動
        monitor_x = monitor_info['left']
        monitor_y = monitor_info['top']
        cv2.moveWindow("Camera Input", monitor_x, monitor_y)
        # ウィンドウサイズをモニターサイズに合わせる
        cv2.resizeWindow("Camera Input", width, height)

    try:
        while True:
            if event.is_set():
                print("terminate camera thread")
                break

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert the frame to PIL Image
            img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Crop the image from the center
            img_width, img_height = img.size
            left_crop = (img_width - width) // 2
            top_crop = (img_height - height) // 2
            right_crop = left_crop + width
            bottom_crop = top_crop + height
            img_cropped = img.crop((left_crop, top_crop, right_crop, bottom_crop))

            # Resize cropped image to fit the monitor window size
            img_resized = img_cropped

            # Display the resized cropped image
            cv2.imshow("Camera Input", cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == ord('q'):
                break

            inputs.append(pil2tensor(img_cropped))

            interval = time.time() - start_time
            fps_interval = 1.0 / UPEER_FPS
            if interval < fps_interval:
                sleep_time = fps_interval - interval
                time.sleep(sleep_time)
    finally:
        cap.release()
        cv2.destroyWindow("Camera Input")
        print('exit : camera')

def read_video(
    event: threading.Event(),
    video_file_path: str,
    height: int = 512,
    width: int = 512,
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
            fps_interval = 1.0 / UPEER_FPS
            if interval < fps_interval:
                sleep_time = fps_interval - interval
                time.sleep(sleep_time)
    finally:
        cap.release()
        cv2.destroyWindow("Video Input")
        print('exit : read_video')

def dummy_screen(
        width: int,
        height: int,
):
    root = tk.Tk()
    root.title("Press Enter to start")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)
    root.attributes("-alpha", 0.8)
    root.configure(bg="black")

    def destroy(event):
        root.destroy()

    def update_geometry(event):
        global top, left
        top = root.winfo_y()
        left = root.winfo_x()

    root.bind("<Return>", destroy)
    root.bind("<Configure>", update_geometry)
    root.mainloop()
    return {"top": top, "left": left, "width": width, "height": height}

def monitor_setting_process(
    width: int,
    height: int,
    monitor_sender: Connection,
) -> None:
    # すべてのモニター情報を取得
    monitors = get_monitors()
    monitor_list = []
    for m in monitors:
        monitor_list.append({
            "id": m.name,
            "width": m.width,
            "height": m.height,
            "left": m.x,
            "top": m.y
        })
    monitor_sender.send(monitor_list)


def apply_gamma_correction(image_np, gamma=2.2):
    return np.power(image_np, 1.0 / gamma)

def normalize_image(image_np):
    # Tensorの最小値と最大値を計算
    min_val = image_np.min()
    max_val = image_np.max()

    # 全体を[0, 1]にスケーリング（min-max正規化）
    normalized_image = (image_np - min_val) / (max_val - min_val)

    return normalized_image

base_img = None


def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    monitor_receiver: Connection,
    engine_dir: Optional[Union[str, Path]],
    prompt_queue,
    video_file_path: Optional[str] = None,
    use_lcm_lora: bool = True,
    use_tiny_vae: bool = True,
    output_dir: str = './output_images',  # 画像保存先フォルダの指定
) -> None:
    """
    画像生成プロセスの関数
    """

    global inputs
    global box_prompt
    instep = 50
    image_index = 0  # 連番用のカウンタを追加

    ######################################################
    # パラメタ
    ######################################################
    adapter = True
    ip_adapter_image_filepath = "assets/xshingoboy-0043.jpg"

    t_index_list = [0, 17, 35]
    cfg_type = "none"
    delta = 1.0

    # Trueで潜在空間の乱数を固定します。
    keep_latent = True

    # fullで有効
    negative_prompt = """(deformed:1.3),(malformed hands:1.4),(poorly drawn hands:1.4),(mutated fingers:1.4),(bad anatomy:1.3),(extra limbs:1.35),(poorly drawn face:1.4),(signature:1.2),(artist name:1.2),(watermark:1.2),(worst quality, low quality, normal quality:1.4), lowres,skin blemishes,extra fingers,fewer fingers,strange fingers,Hand grip,(lean),Strange eyes,(three arms),(Many arms),(watermarking)"""
    ######################################################

    # フォルダの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ControlNetモデルの準備
    controlnet_pose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16
    ).to("cuda")

    # ipAdapterのイメージエンコーダ
    image_encoder = None
    if adapter:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Lykon/dreamshaper-8-lcm",
        controlnet=controlnet_pose,
        image_encoder=image_encoder).to(
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

    if adapter:
        pipe.load_ip_adapter('h94/IP-Adapter', subfolder="models",
                             weight_name="ip-adapter_sd15.bin",
                             torch_dtype=torch.float16)
        pipe.set_ip_adapter_scale(0.8)

    pipe.load_lora_weights("./models/LoRA/xshingogirl.safetensors", adapter_name="xshingogirl")
    pipe.set_adapters(["xshingogirl"], adapter_weights=[1.0])

    # Diffusers pipelineをStreamDiffusionにラップ
    stream = StreamDiffusionControlNetSample(
        pipe,
        t_index_list=t_index_list,
        torch_dtype=torch.float16,
        cfg_type=cfg_type,
        width=width,
        height=height,
        ip_adapter=adapter,
        acceleration=acceleration,
        model_id_or_path=model_id_or_path,
    )

    # Tiny VAEで高速化
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    ip_adapter_image = None
    if adapter:
        print("prepare ip adapter")
        ip_adapter_image = load_image(ip_adapter_image_filepath)

    stream.prepare(
        prompt=box_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=instep,
        guidance_scale=guidance_scale,
        delta=delta,
        ip_adapter_image=ip_adapter_image
    )

    # カメラが接続されているか確認
    if video_file_path is not None:
        input_source = 'video'
    else:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            input_source = 'camera'
            cap.release()
        else:
            input_source = 'screen'

    monitor_list = monitor_receiver.recv()
    if len(monitor_list) > 1:
        monitor_info = monitor_list[1]  # 2番目のモニター
    else:
        monitor_info = monitor_list[0]  # メインモニター

    event = threading.Event()
    event.clear()

    if input_source == 'camera':
        input_thread = threading.Thread(target=camera, args=(event, height, width, monitor_info))
    elif input_source == 'video':
        input_thread = threading.Thread(target=read_video, args=(event, video_file_path, height, width, monitor_info))
    else:
        input_thread = threading.Thread(target=screen, args=(event, height, width, monitor_info))

    input_thread.start()
    time.sleep(1)
    current_prompt = box_prompt
    total_frames=0

    while True:
        try:

            if not close_queue.empty():  # close_queueに終了信号が来たら停止
                print("Termination signal received.")
                break
            if len(inputs) < frame_buffer_size:
                time.sleep(fps_interval)
                continue
            start_time = time.time()
            sampled_inputs = []
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            new_prompt = current_prompt
            if not prompt_queue.empty():
                new_prompt = prompt_queue.get(block=False)

            prompt_change = False
            if current_prompt != new_prompt:
                current_prompt = new_prompt
                stream.update_prompt(current_prompt, negative_prompt)
                prompt_change = True

            input = input_batch.to(device=stream.device, dtype=stream.dtype)
            global base_img
            if base_img is None:
                base_img = input

            output_images = stream.ctlimg2img(ctlnet_image=input, keep_latent=keep_latent)

            if frame_buffer_size == 1:
                output_images = [output_images]
            total_frames+=len(output_images)
            print(f"total {total_frames}", end='', flush=True)

            for output_image in output_images:
                queue.put(output_image, block=False)
                
                # output_imageがtorch.Tensor型の場合、変換
                if isinstance(output_image, torch.Tensor):
                    # データをCPUに移動してnumpyに変換
                    output_image_np = output_image.squeeze().cpu().numpy()

                    # 値の範囲を確認
                    #print(f"Tensorの値の範囲: min={output_image_np.min()}, max={output_image_np.max()}")

                    # 値を正規化して、[0, 1]の範囲にスケーリング
                    output_image_np = normalize_image(output_image_np)

                    # [0, 255]にスケーリングし、uint8にキャスト
                    output_image_np = np.clip(output_image_np * 255, 0, 255).astype(np.uint8)

                    # チャンネルの順序を調整 (C, H, W) -> (H, W, C)
                    if output_image_np.ndim == 3 and output_image_np.shape[0] == 3:  # RGB画像の場合
                        output_image_np = np.moveaxis(output_image_np, 0, -1)

                    # 変換後の値の範囲を再確認
                    #print(f"画像データの変換後の範囲: min={output_image_np.min()}, max={output_image_np.max()}")

                    # PIL Imageに変換
                    output_pil_image = Image.fromarray(output_image_np)

                    # 画像を保存
                    image_index += 1  # 連番のインデックスを更新
                    output_image_path = os.path.join(output_dir, f"output_image_{image_index:04d}.png")
                    output_pil_image.save(output_image_path)  # PNG形式で保存

            process_time = time.time() - start_time
            if process_time <= fps_interval:
                time.sleep(fps_interval - process_time)
            process_time = time.time() - start_time
            fps = 1 / (process_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            break
        except:
            print("An error occurred.")
            break


def main(
    model_id_or_path: str = "Lykon/dreamshaper-8-lcm",
    lora_dict: Optional[Dict[str, float]] = {"./models/LoRA/xshingogirl.safetensors": 0.9},
    prompt: str = "xshingogirl",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 960,
    height: int = 540,
    acceleration: Literal["none", "xformers", "tensorrt"] = "none",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "none",
    guidance_scale: float = 1.4,
    delta: float = 0.5,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
    engine_dir: Optional[Union[str, Path]] = "engines",
    video_file_path: Optional[str] = "./assets/mptest.mp4",
) -> None:
    """
    メイン関数
    """

    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    prompt_queue = ctx.Queue()
    close_queue = Queue()

    do_add_noise = False
    monitor_sender, monitor_receiver = ctx.Pipe()

    # prompt_process = ctx.Process(
    #     target=prompt_window,
    #     args=(
    #         prompt_queue,
    #     ),
    # )
    # prompt_process.start()

    process1 = ctx.Process(
        target=image_generation_process,
        args=(
            queue,
            fps_queue,
            close_queue,
            model_id_or_path,
            lora_dict,
            prompt,
            negative_prompt,
            frame_buffer_size,
            width,
            height,
            acceleration,
            use_denoising_batch,
            seed,
            cfg_type,
            guidance_scale,
            delta,
            do_add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            monitor_receiver,
            engine_dir,
            prompt_queue,
            video_file_path  # 追加
        ),
    )
    process1.start()

    monitor_process = ctx.Process(
        target=monitor_setting_process,
        args=(
            width,
            height,
            monitor_sender,
        ),
    )
    monitor_process.start()
    monitor_process.join()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    # terminate
    process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    process1.join(5)  # with timeout
    if process1.is_alive():
        print("process1 still alive. force killing...")
        process1.terminate()  # force kill...
    process1.join()
    print("process1 terminated.")


if __name__ == "__main__":
    fire.Fire(main)
