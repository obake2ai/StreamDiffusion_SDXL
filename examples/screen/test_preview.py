import cv2
import threading
import time
import torch
import PIL.Image
from multiprocessing import Process, Queue, get_context
from streamdiffusion.image_utils import pil2tensor

def dummy_image_generation_process(queue, result_queue):
    while True:
        try:
            frame_tensor = queue.get()
            if frame_tensor is None:  # プロセス終了のシグナル
                break

            # ダミーの画像生成処理
            result_queue.put(frame_tensor)
        except Exception as e:
            print(f"Error in dummy_image_generation_process: {e}")
            break

    print("Exiting dummy_image_generation_process")

def preview_video_with_generation(video_path):
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)

    # 動画が開けなかった場合のエラーチェック
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 生成プロセスの設定
    ctx = get_context('spawn')
    queue = ctx.Queue()
    result_queue = ctx.Queue()

    process = ctx.Process(
        target=dummy_image_generation_process,
        args=(queue, result_queue)
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
        if not result_queue.empty():
            output_image = result_queue.get()
            output_image = output_image.permute(1, 2, 0).numpy()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Generated Video Preview', output_image)

        # 'q' キーが押されたら終了
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()
    queue.put(None)  # プロセス終了シグナル
    process.join()   # プロセスの終了を待機

if __name__ == '__main__':
    # テスト用の動画ファイルのパスを指定
    video_path = 'assets/0710_MPtestsozai.mp4'
    preview_video_with_generation(video_path)
