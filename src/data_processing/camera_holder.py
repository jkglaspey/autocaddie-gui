import cv2
from PIL import Image, ImageTk

class CameraHolder:
    def __init__(self, path):
        self.path = path
        self.cap = None
        self.width = None
        self.height = None
        self.fps = None
        self.frames = None
        self.numFrames = None
        
    def open_camera(self):
        self.cap = cv2.VideoCapture(self.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 1
        self.convert_to_frames()
        self.close_camera()

    def convert_to_frames(self):
        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_image = Image.fromarray(opencv_image)
            self.frames.append(captured_image)
        self.numFrames = len(self.frames)

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def set_dims(self, width, height):
        self.width = width
        self.height = height

    def getFrame(self, idx):
        if self.frames is not None and idx < self.numFrames and idx >= 0 and self.width is not None and self.height is not None:
            return self.resize_and_return(idx)
        
        else:
            return None, None

    def resize_and_return(self, idx):
        resized_image = self.frames[idx].resize((self.width, self.height))
        photo = ImageTk.PhotoImage(image = resized_image)
        return resized_image, photo
