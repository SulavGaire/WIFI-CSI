import cv2

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {int(total_frames)}")
    print(f"Video Duration: {duration:.2f} seconds")


get_video_fps('/home/noha/Documents/CSIGenAI/dataset/Chair/videos/20250807_090604_859.webm')
