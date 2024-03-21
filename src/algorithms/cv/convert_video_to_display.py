import cv2
import os
import shutil
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

def close_camera(camera):
    camera.release()

# Never called
def update_cameras(cameras, width, height):
    for camera in cameras:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

global images

def reset_images_folder():
    global images
    images = []
    #directory_path = "camera_data"
    #try:
    #    shutil.rmtree(directory_path)
    #except Exception as e:
    #    print(f"Error deleting directory '{directory_path}': {e}")

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
    cam = 0
    pathname = r"data_processing\camera_data"
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    for i, image in enumerate(images):
        print(f"Saving image! {(i//2)}")
        image_filename = os.path.join(pathname, f"{cam}-{(i//2)}.png")
        cv2.imwrite(image_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cam = 1 - cam
