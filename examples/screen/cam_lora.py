import os
import sys
import time
import threading  # threadingモジュールのインポートを追加
from multiprocessing import Process, Queue, get_context
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
from streamdiffusion.image_utils import pil2tensor
import mss
import fire
import tkinter as tk
import cv2  # OpenCVを使用してカメラからの画像を取得

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

inputs = []
top = 0
left = 0

def video_feed(event: threading.Event, video_path: str, height: int = 512, width: int = 512, convert_to_grayscale: bool = False):
    """
    動画から画像を取得し、ストリーミングする関数
    """
    global inputs
    cap = cv2.VideoCapture(video_path)  # 動画ファイルを指定

    while True:
        if event.is_set():
            print("terminate video feed thread")
            break
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame or end of video reached")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 動画の再生が終わったら最初に戻る
            continue

        if convert_to_grayscale:
            # グレースケールに変換してから3チャンネルの画像に戻す
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((width, height))  # widthとheightをカメラの解像度に合わせてリサイズ
        inputs.append(pil2tensor(img))

    cap.release()
    print('exit: video_feed')


def camera_feed(event: threading.Event, height: int = 512, width: int = 512, convert_to_grayscale: bool = False, video_path: Optional[str] = None):
    """
    カメラから画像を取得し、ストリーミングする関数
    """
    global inputs
    cap = cv2.VideoCapture(0)  # /dev/video0に対応するカメラデバイスを指定
    if not cap.isOpened() and video_path:
        print("No camera found. Falling back to video file.")
        video_feed(event, video_path, height, width, convert_to_grayscale)
        return

    while True:
        if event.is_set():
            print("terminate camera feed thread")
            break
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break

        if convert_to_grayscale:
            # グレースケールに変換してから3チャンネルの画像に戻す
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((width, height))  # widthとheightをカメラの解像度に合わせてリサイズ
        inputs.append(pil2tensor(img))

    cap.release()
    print('exit: camera_feed')


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
    root.bind("<Return>", destroy)
    def update_geometry(event):
        global top, left
        top = root.winfo_y()
        left = root.winfo_x()
    root.bind("<Configure>", update_geometry)
    root.mainloop()
    return {"top": top, "left": left, "width": width, "height": height}

def monitor_setting_process(
    width: int,
    height: int,
    monitor_sender: Connection,
) -> None:
    monitor = dummy_screen(width, height)
    monitor_sender.send(monitor)

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
    monitor_receiver : Connection,
) -> None:
    """
    画像を生成するプロセス
    """
    global inputs
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,  # widthを正しく指定
        height=height,  # heightを正しく指定
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    monitor = monitor_receiver.recv()

    event = threading.Event()
    input_screen = threading.Thread(target=camera_feed, args=(event, height, width, True))  # グレースケールを有効にする
    input_screen.start()
    time.sleep(5)

    while True:
        try:
            if not close_queue.empty(): # 終了の確認
                break
            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue
            start_time = time.time()
            sampled_inputs = []
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            output_images = stream.stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu()
            if frame_buffer_size == 1:
                output_images = [output_images]
            for output_image in output_images:
                queue.put(output_image, block=False)

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            break

    print("closing image_generation_process...")
    event.set() # キャプチャスレッドを停止
    input_screen.join()
    print(f"fps: {fps}")

def main(
    model_id_or_path: str = "Lykon/AnyLoRA",
    lora_dict: Optional[Dict[str, float]] = {"./models/LoRA/xshingoboy.safetensors":1.0},
    prompt: str = "xshingoboy",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 1280,  # 高解像度に設定
    height: int = 720,  # 高解像度に設定
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    guidance_scale: float = 1.4,
    delta: float = 0.5,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
    video_path: Optional[str] = None,  # 動画ファイルのパスを追加
) -> None:
    """
    メイン関数。画像生成とビューアープロセスを開始します。
    """
    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    close_queue = Queue()

    monitor_sender, monitor_receiver = ctx.Pipe()

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

    event = threading.Event()
    input_screen = threading.Thread(target=camera_feed, args=(event, height, width, True, video_path))  # video_pathを渡す
    input_screen.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    # 終了処理
    process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    process1.join(5) # タイムアウト付き
    if process1.is_alive():
        print("process1 still alive. force killing...")
        process1.terminate() # 強制終了
    process1.join()
    print("process1 terminated.")
    event.set()  # キャプチャスレッドを停止
    input_screen.join()


if __name__ == "__main__":
    fire.Fire(main)
