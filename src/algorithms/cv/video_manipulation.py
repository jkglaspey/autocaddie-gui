# Define the CV algorithms to deal with video data

import cv2
import os
import numpy as np
import time

def convert_frames_to_video():
    pathname = r"data_processing\camera_data"
    fps = 60
    time_between_frames = 1.0 / fps

    # Create a list of frame paths for each camera
    cam_0_frames = sorted([os.path.join(pathname, f) for f in os.listdir(pathname) if f.startswith('0-')])
    cam_1_frames = sorted([os.path.join(pathname, f) for f in os.listdir(pathname) if f.startswith('1-')])

    # Get the dimensions of the first frame
    first_frame_0 = cv2.imread(cam_0_frames[0])
    height_0, width_0, _ = first_frame_0.shape
    first_frame_1 = cv2.imread(cam_1_frames[0])
    height_1, width_1, _ = first_frame_1.shape

    # Create video writers for each camera
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cam_0_out = cv2.VideoWriter(r"data_processing\video_data\cam_0_in.mp4", fourcc, fps, (width_0, height_0))
    cam_1_out = cv2.VideoWriter(r"data_processing\video_data\cam_1_in.mp4", fourcc, fps, (width_1, height_1))

    start_time = time.time()  # Get the start time
    for frame_path in cam_0_frames:
        frame = cv2.imread(frame_path)
        cam_0_out.write(frame)
        elapsed_time = time.time() - start_time  # Calculate the elapsed time
        remaining_time = time_between_frames - elapsed_time  # Calculate the remaining time
        if remaining_time > 0:
            time.sleep(remaining_time)  # Wait for the remaining time
        start_time += time_between_frames  # Update the start time for the next frame

    start_time = time.time()  # Reset the start time for the second camera
    for frame_path in cam_1_frames:
        frame = cv2.imread(frame_path)
        cam_1_out.write(frame)
        elapsed_time = time.time() - start_time
        remaining_time = time_between_frames - elapsed_time
        if remaining_time > 0:
            time.sleep(remaining_time)
        start_time += time_between_frames

    # Release the video writers
    cam_0_out.release()
    cam_1_out.release()

if __name__ == "__main__":
    convert_frames_to_video()
