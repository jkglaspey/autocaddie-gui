import cv2
import threading
import time
from PIL import Image, ImageTk

class CameraRecorder(threading.Thread):
    def __init__(self, camera, output_files, width, height, idx):
        super().__init__()
        self.camera = camera
        self.output_files = output_files
        self.is_running = False
        self.save = False
        self.width = width
        self.height = height
        self.idx = idx
    
    def keep_reference(self, window, canvas, rectangle):
        self.window = window
        self.canvas = canvas
        self.rectangle = rectangle

    def change_dims_of_self(self, width, height):
        self.width = width
        self.height = height

    def set_recording_to_true(self):
        self.save = True

    def run(self):
        recorder, video_writer = self.start_recording(self.camera, self.output_files)
        self.video_writer = video_writer
        self.is_running = True
        while self.is_running and not self.save:
            ret, frame = recorder.read()
            if ret:
                self.send_frame_to_gui(frame, self.idx)
        while self.is_running and self.save:
            ret, frame = recorder.read()
            if ret:
                self.send_frame_to_gui(frame, self.idx)
                video_writer.write(frame)
            else:
                print(f"Missed a frame on cam #{self.idx}")
        self.stop_recording()

    def start_recording(self, recorder, output_file):
        fps = recorder.get(cv2.CAP_PROP_FPS)
        width = int(recorder.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(recorder.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        return recorder, out

    def stop_recording(self):
        self.is_running = False
        self.save = False
        self.camera.release()
        self.video_writer.release()
    
    def send_frame_to_gui(self, frame, idx):
        from gui_module.build.gui_recording import update_camera_feed
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        captured_image = Image.fromarray(opencv_image)
        resized_image = captured_image.resize((self.width, self.height))
        photo = ImageTk.PhotoImage(image=resized_image)
        update_camera_feed(photo, self.canvas, self.window, self.width, self.height, self.rectangle, self.idx)

def start_recording(cameras, output_files, width, height, idx):
    recorder_thread = CameraRecorder(cameras, output_files, width, height, idx)
    recorder_thread.start()
    return recorder_thread

def change_dims(recorder_thread, width, height):
    recorder_thread.change_dims_of_self(width, height)

def stop_recording(recorder_thread):
    recorder_thread.stop_recording()


if __name__ == "__main__":
    pass