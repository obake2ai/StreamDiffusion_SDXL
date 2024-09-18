import cv2

def preview_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 元の動画のフレームレートを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24  # フレームレートが取得できない場合のデフォルト値
        print("Warning: Could not retrieve FPS. Defaulting to 30 FPS.")

    frame_delay = int(1000 / fps)  # フレーム間の待機時間（ミリ秒単位）

    while True:
        ret, frame = cap.read()
        if not ret:
            # 動画の終端に達したら最初に戻る
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow('Video Preview', frame)

        # 'q'キーを押すと再生を終了
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 再生する動画ファイルのパスを指定
    video_path = 'assets/0710_MPtestsozai.mp4'
    preview_video(video_path)
