import cv2
from PIL import Image, ImageTk

def capture_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    _, frame = cap.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    else :
        print("Failed to capture a frame")

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

    
# This function doesn't work
def update_video(canvas, camera, rectangle_id, canvas_item_id, width, height, x, y):
    _, frame = camera.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(opencv_image)
    photo = ImageTk.PhotoImage(image=captured_image)
    canvas.itemconfig(rectangle_id, image=photo)
    canvas_item_id = photo  # Prevent garbage collection
    canvas.coords(rectangle_id, x, y)

def get_frame_from_camera(camera, width, height):
    _, frame = camera.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(opencv_image)
    resized_image = captured_image.resize((width, height))
    photo = ImageTk.PhotoImage(image=captured_image)
    return photo