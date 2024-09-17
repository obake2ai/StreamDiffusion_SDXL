import os
import sys
import cv2
import threading
import time
import torch
import PIL.Image
from multiprocessing import Process, Queue, get_context
from streamdiffusion.image_utils import pil2tensor

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

def image_generation_process(queue, model_id_or_path, prompt, negative_prompt, width, height):
    # StreamDiffusionWrapper の設定
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=None,  # 必要に応じて指定
        t_index_list=[32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration="xformers",
        do_add_noise=False,
        enable_similar_image_filter=False,
        similar_image_filter_threshold=0.99,
        similar_image_filter_max_skip_frame=10,
        mode="img2img",
        use_denoising_batch=True,
        cfg_type="self",
        seed=2,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.4,
        delta=0.5,
    )

    while True:
        try:
            frame_tensor = queue.get()
            input_batch = frame_tensor.unsqueeze(0).to(device=stream.device, dtype=stream.dtype)
            output_image = stream.stream(input_batch)[0].cpu()
            queue.put(output_image)
        except Exception as e:
            print(f"Error in image_generation_process: {e}")
            break

    print("Exiting image_generation_process")

def preview_video_with_generation(video_path, model_id_or_path, prompt, negative_prompt):
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)

    # 動画が開けなかった場合のエラーチェック
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 生成プロセスの設定
    ctx = get_context('spawn')
    queue = ctx.Queue()

    process = ctx.Process(
        target=image_generation_process,
        args=(queue, model_id_or_path, prompt, negative_prompt, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    process.start()

    # 動画のフレームを1つずつ表示
    while True:
        ret, frame = cap.read()

        # フレームが読み込めなかった場合はループを終了
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # フレームを PIL.Image に変換して tensor に変換
        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = pil2tensor(img)

        # フレームを生成プロセスに渡す
        queue.put(frame_tensor)

        # 生成された画像を取得
        if not queue.empty():
            output_image = queue.get()
            output_image = output_image.permute(1, 2, 0).numpy()  # バッチ次元を取り除く
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Generated Video Preview', output_image)

        # 'q' キーが押されたら終了
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()
    process.terminate()

if __name__ == '__main__':
    # テスト用の動画ファイルのパスを指定
    video_path = 'assets/0710_MPtestsozai.mp4'
    model_id_or_path = "Lykon/AnyLoRA"  # モデルIDまたはパス
    prompt = "xshingoboy"
    negative_prompt = "low quality, bad quality, blurry, low resolution"

    preview_video_with_generation(video_path, model_id_or_path, prompt, negative_prompt)
