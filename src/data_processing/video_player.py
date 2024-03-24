import moviepy.editor as mp
import cv2
from PIL import Image, ImageTk

class VideoPlayer():

    def __init__(self, path, canvas = None, window = None, rectangle = None, width = None, height = None, idx = 0):
        self.path = path
        self.canvas = canvas
        self.window = window
        self.rectangle = rectangle
        self.width = width
        self.height = height
        self.idx = idx
        self.is_playing = False

    def change_dims(self, width, height):
        self.width = width
        self.height = height

    def play_clip_on_loop(self):
        # Load the video clip
        clip = mp.VideoFileClip(self.path)

        cropped_clip = clip.subclip(0, self.trim)

        # Infinite loop until 'q' is pressed
        self.is_playing = True
        while self.is_playing:
            # Play the video
            for frame in cropped_clip.iter_frames():
                if not self.is_playing:
                    return

                # Convert frame color space from RGB to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Prep it to display to GUI
                self.prep_frame_to_send(frame_bgr)

                # Show it!
                #cv2.imshow("Video", frame_bgr)

                # Wait for a key press for a duration based on the frame rate
                key = cv2.waitKey(int(1000 / clip.fps))

                # Check if 'q' key was pressed or the wait timed out
                # Only use for debugging purposes...
                #if key == ord("q"):
                #    self.is_playing = False
                #    break

        # Close the OpenCV window
        #cv2.destroyAllWindows()
    
    def prep_frame_to_send(self, frame):
        from gui_module.build.gui_generating_results import send_image_to_update_canvas_image
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        captured_image = Image.fromarray(opencv_image)
        self.image = captured_image
        resized_image = captured_image.resize((self.width, self.height))
        photo = ImageTk.PhotoImage(image=resized_image)
        send_image_to_update_canvas_image(photo, self)

    def get_collection_of_frames(self):
        frames = []
        clip = mp.VideoFileClip(self.path)
        for frame in clip.iter_frames():
            frames.append(frame)
        return frames
    
    def get_image_and_photo(self, frame):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        opencv_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        captured_image = Image.fromarray(opencv_image)
        resized_image = captured_image.resize((self.width, self.height))
        photo = ImageTk.PhotoImage(image=resized_image)
        return captured_image, photo 

    def convert_frames_to_video(self, path, frames):
        this_clip = mp.VideoFileClip(self.path)
        clip = mp.ImageSequenceClip(frames, fps=this_clip.fps)
        clip.write_videofile(path, codec='libx264')

    def get_clip_length(self):
        return mp.VideoFileClip(self.path).duration
    
    def set_trim_length(self, length):
        self.trim = length

    def stop_playing(self):
        self.is_playing = False

if __name__ == "__main__":
    video_player = VideoPlayer(r"data_processing\video_data\output_1.mp4")
    video_player.play_clip_on_loop()
