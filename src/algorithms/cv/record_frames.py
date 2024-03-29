import cv2
from find_valid_camera_indices import find_camera_indices

def extract_frames(camera_indices, duration=30):

    # Initialize lists for each camera
    cameras = [cv2.VideoCapture(idx, cv2.CAP_DSHOW) for idx in camera_indices]
    frames = [[] for _ in camera_indices]
    
    # Record for the specified duration
    end_time = cv2.getTickCount() + duration * cv2.getTickFrequency()
    count = 0
    while cv2.getTickCount() < end_time:
        #count += 1
        #print(count)
        ret_frames = [camera.read() for camera in cameras]

        # Bad frame? Skip it
        if not all(ret_frames):
            break

        # Add frame to collection
        for i, (ret, frame) in enumerate(ret_frames):
            if ret:
                frames[i].append(frame)

    # Release cameras
    for camera in cameras:
        camera.release()

    return frames

if __name__ == "__main__":
    # Specify your camera indices here
    camera_indices = find_camera_indices()

    recorded_frames = extract_frames(camera_indices)

    # Save frames to the "out" folder
    total = 0
    for i, frames in enumerate(recorded_frames):
        for j, frame in enumerate(frames):
            success = cv2.imwrite(f"algorithms/cv/out/camera_{i}_frame_{j}.png", frame)
