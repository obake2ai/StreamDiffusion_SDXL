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
            if frame_tensor is None:  # Process termination signal
                break

            # Dummy image generation processing
            result_queue.put(frame_tensor)
        except Exception as e:
            print(f"Error in dummy_image_generation_process: {e}")
            break

    print("Exiting dummy_image_generation_process")

def preview_video_with_generation(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Error check if the video cannot be opened
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Setup the generation process
    ctx = get_context('spawn')
    queue = ctx.Queue()
    result_queue = ctx.Queue()

    process = ctx.Process(
        target=dummy_image_generation_process,
        args=(queue, result_queue)
    )
    process.start()

    # Iterate over video frames
    while True:
        ret, frame = cap.read()

        # End loop if frame cannot be read
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # Convert frame to PIL.Image and then to tensor
        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = pil2tensor(img)

        # Send frame to generation process
        queue.put(frame_tensor)

        # Retrieve generated image
        if not result_queue.empty():
            output_image = result_queue.get()
            output_image = output_image.permute(1, 2, 0).numpy()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            # Instead of imshow, write the frame to a file or process it further
            # For example, save the frame as an image file
            cv2.imwrite('output_frame.jpg', output_image)

        # Remove or adjust the waitKey if not displaying the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    queue.put(None)  # Process termination signal
    process.join()   # Wait for the process to finish

if __name__ == '__main__':
    # Specify the path to the test video file
    video_path = 'assets/0710_MPtestsozai.mp4'
    preview_video_with_generation(video_path)
