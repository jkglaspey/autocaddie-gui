import cv2
from PIL import Image, ImageTk

def open_cameras(camera_indices):
    cameras = []
    for idx in camera_indices:
        camera = cv2.VideoCapture(idx)
        if camera.isOpened():
            cameras.append(camera)
        else:
            print(f"Failed to open camera at index {idx}")
    return cameras

def close_cameras(cameras):
    for camera in cameras:
        camera.release()

# Never called
def update_cameras(cameras, width, height):
    for camera in cameras:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def get_frame_from_camera(camera, width, height):
    _, frame = camera.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(opencv_image)
    resized_image = captured_image.resize((width, height))
    photo = ImageTk.PhotoImage(image=resized_image)
    return photo