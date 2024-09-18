import cv2

def preview_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("動画の終端に到達しました。")
            break

        cv2.imshow('Video Preview', frame)

        # 'q'キーを押すと再生を終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 再生する動画ファイルのパスを指定
    video_path = 'assets/0710_MPtestsozai.mp4'
    preview_video(video_path)
