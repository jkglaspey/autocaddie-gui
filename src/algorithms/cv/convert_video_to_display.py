import cv2
import os
import shutil
from PIL import Image, ImageTk

def open_cameras(camera_indices):
    cameras = []
    for idx in camera_indices:
        camera = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FPS, 60)
            cameras.append(camera)
        else:
            print(f"Failed to open camera at index {idx}")
    return cameras

def close_cameras(cameras):
    for camera in cameras:
        camera.release()

def close_camera(camera):
    camera.release()

# Never called
def update_cameras(cameras, width, height):
    for camera in cameras:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

global images
images = []
def reset_images_folder():
    global images
    images = []
    directory_path = r"data_processing/camera_data"
    try:
        os.system(f"rmdir /S /Q \"{directory_path}\"")
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        print(f"Error deleting directory '{directory_path}': {e}")

def get_frame_from_camera(camera, width, height, save_image=False):
    _, frame = camera.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(opencv_image)
    resized_image = captured_image.resize((width, height))
    photo = ImageTk.PhotoImage(image=resized_image)

    if save_image == True:
        images.append(opencv_image)

    # Save the image to storage?
    return photo

def save_images_to_folder():
    global images
    cam = 0
    pathname = r"data_processing\camera_data"
    if not os.path.exists(pathname):
        os.makedirs(pathname)

    pads = "00"
    for i, image in enumerate(images):
        if i == 20:
            pads == "0"
        elif i == 200:
            pads == ""
        image_filename = os.path.join(pathname, f"{cam}-{pads}{(i//2)}.png")
        cv2.imwrite(image_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cam = 1 - cam

    images = []
