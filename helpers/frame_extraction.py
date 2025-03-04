import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, frame_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
            
        if frame_count % int(fps * frame_interval) == 0:
            frame_path = os.path.join(output_dir, f"{Path(video_path).stem}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            
        frame_count += 1
    
    video.release()
    return output_dir