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
    video_path: Optional[str] = "assets/0710_MPtestsozai.mp4",  # 動画ファイルのパスを追加
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

    # terminate
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
