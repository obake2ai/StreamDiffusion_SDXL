import cv2

def preview_video(video_path):
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)

    # 動画が開けなかった場合のエラーチェック
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 動画のフレームを1つずつ表示
    while True:
        ret, frame = cap.read()

        # フレームが読み込めなかった場合はループを終了
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # フレームを表示
        cv2.imshow('Video Preview', frame)

        # 'q' キーが押されたら終了
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

# テスト用の動画ファイルのパスを指定
video_path = 'assets/0710_MPtestsozai.mp4'
preview_video(video_path)
