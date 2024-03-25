
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
import json
import threading
import time
from algorithms.cv.convert_video_to_display import save_images_to_folder, reset_images_folder
from algorithms.cv.video_manipulation import convert_frames_to_video
from PIL import Image, ImageTk
from data_processing.video_player import VideoPlayer
from neural_network.process_videos import execute_process_video

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, ttk, Canvas, Entry, Text, Button, PhotoImage
import tkinter as tk


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r".\assets\frame7")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def save_window_state(width, height, fullscreen, x, y, state):
    with open(r"gui_module\build\assets\window_state.json", "w") as f:
        json.dump({"width": width, "height": height, "fullscreen": fullscreen, "x": x, "y": y, "maximized": state}, f)

def load_window_state():
    try:
        with open(r"gui_module\build\assets\window_state.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    
def create_window(width, height, fullscreen, x, y, maximized):
    window = Tk()
    window.geometry(f"{width}x{height}")
    window.configure(bg="#FFFFFF")
    window.geometry("+{}+{}".format(x, y))

    # Maximized or fullscreen?
    if maximized:
        window.state('zoomed')
    if fullscreen:
        window.attributes("-fullscreen", True)
    
    # Save window state before closing
    window.protocol("WM_DELETE_WINDOW", lambda: close_window(window, width, height, x, y))
    
    return window

def close_window(window, width, height, x, y):
    if window.attributes("-fullscreen") == 0:
            x, y = get_window_position(window)
    save_window_state(width, height, window.attributes("-fullscreen"), x, y, window.state() == 'zoomed')
    global terminate_early
    terminate_early = True
    window.destroy()

def get_window_position(window):
    geometry_string = window.geometry()
    x, y = map(int, geometry_string.split('+')[1:])
    return x, y

def window_event(window):
    if window.state() != 'zoomed':
        return window.winfo_width(), window.winfo_height()
    else:
        return 0, 0
    
def send_image_to_update_canvas_image(frame, video_player):
    video_player.window.after(0, lambda: update_canvas_image(frame, video_player))
    
def update_canvas_image(frame, video_player):
    global canvas_width, canvas_height
    if video_player.idx == 0:
        video_player.canvas.itemconfig(video_player.rectangle, image=frame)
        video_player.canvas.rectangle_2 = frame
        video_player.canvas.rectangle_image_2 = video_player.image
        video_player.canvas.coords(video_player.rectangle, canvas_width * 0.020, canvas_height * 0.429)
    else:
        video_player.canvas.itemconfig(video_player.rectangle, image=frame)
        video_player.canvas.rectangle_3 = frame
        video_player.canvas.rectangle_image_3 = video_player.image
        video_player.canvas.coords(video_player.rectangle, canvas_width * 0.517, canvas_height * 0.429)

def main():
    print("Entered main!")
    saved_state = load_window_state()
    width = 0
    height = 0
    x = 0
    y = 0
    if saved_state:
        width, height, fullscreen, x, y, maximized = saved_state["width"], saved_state["height"], saved_state["fullscreen"], saved_state["x"], saved_state["y"], saved_state["maximized"]
    else:
        width, height, fullscreen, x, y, maximized = 797, 448, False, 0, 0, False
    
    window = create_window(width, height, fullscreen, x, y, maximized)

    # Fullscreen
    def toggle_fullscreen(event=None):
        state = not window.attributes("-fullscreen")
        window.attributes("-fullscreen", state)
        return "break"

    def end_fullscreen(event=None):
        window.attributes("-fullscreen", False)
        return "break"

    # Bind F11 key to toggle fullscreen
    window.bind("<F11>", toggle_fullscreen)
    # Bind Escape key to end fullscreen
    window.bind("<Escape>", end_fullscreen)

    # Body
    canvas = Canvas(
        window,
        bg = "#FFFFFF",
        height = 448,
        width = 797,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )
    canvas.pack(fill="both", expand=True)

    # Initialize dimensions
    global camera_width, camera_height, canvas_width, canvas_height
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    camera_width = (int)(canvas_width * 0.464)
    camera_height = (int)(canvas_height * 0.535)
    if camera_width == 0:
        camera_width = 1
    if camera_height == 0:
        camera_height = 1

    # Globals
    global terminate_early
    terminate_early = False
    
    # Image 1
    global image_1_gr
    canvas.image_1_gr = None
    image_image_1_gr = Image.open(relative_to_assets("bk_image_1.png"))
    photo_image_1_gr = ImageTk.PhotoImage(image_image_1_gr)
    canvas.image_1_gr = photo_image_1_gr
    image_1_gr = canvas.create_image(0, 0, anchor="nw", image=photo_image_1_gr)

    # Objects to be created later
    rectangle_1 = None
    text_1 = None
    rectangle_4 = None
    text_4 = None
    rectangle_5 = None
    text_5 = None
    button_1 = None
    button_2 = None
    button_3 = None
    window.button_image_1 = None
    window.button_image_2 = None
    window.button_image_3 = None
    global frame_idx_var
    frame_idx_var = 0
    slider = None

    # Placeholders for future video streams
    dark_black_image = Image.new("RGB", (370, 240), "#1E1E1E")
    photo_image_dark_black = ImageTk.PhotoImage(dark_black_image)
    rectangle_2 = canvas.create_image(16, 192, anchor="nw", image=photo_image_dark_black)
    canvas.rectangle_2 = photo_image_dark_black
    canvas.rectangle_image_2 = dark_black_image
    rectangle_3 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
    canvas.rectangle_3 = photo_image_dark_black
    canvas.rectangle_image_3 = dark_black_image

    # Resize background image
    def resize_background(event=None):

        # Resize the image to fit the canvas size
        resized_image_1_gr = image_image_1_gr.resize((canvas_width, canvas_height))

        # Convert the resized image to a Tkinter-compatible format
        photo_image_1_gr = ImageTk.PhotoImage(resized_image_1_gr)

        # Update the canvas with the resized image
        canvas.itemconfig(image_1_gr, image=photo_image_1_gr)
        canvas.image_1_gr = photo_image_1_gr  # Keep a reference to prevent garbage collection
    
    # Resize elements in window
    def resize_canvas(event=None):

        # Window
        temp_width, temp_height = window_event(window)
        if temp_width > 0 and temp_height > 0:
            width = temp_width
            height = temp_height
        if window.attributes("-fullscreen") == 0:
            x, y = get_window_position(window)

        # Get the size of the canvas
        global canvas_width, canvas_height
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Resize the background image
        resize_background()

        # Rectangles
        nonlocal rectangle_1, rectangle_4, rectangle_5
        if rectangle_1:
            new_cords = [canvas_width * 0.068, canvas_height * 0.098, canvas_width * 0.444, canvas_height * 0.310]
            canvas.coords(rectangle_1, *new_cords)
        if rectangle_4:
            new_cords = [canvas_width * 0.068, canvas_height * 0.098, canvas_width * 0.444, canvas_height * 0.310]
            canvas.coords(rectangle_4, *new_cords)
        if rectangle_5:
            new_cords = [canvas_width * 0.344, canvas_height * 0.464, canvas_width * 0.720, canvas_height * 0.536]
            canvas.coords(rectangle_5, *new_cords)

        # Text
        nonlocal text_1, text_4, text_5
        if text_1:
            text_x = canvas_width / 18.667
            text_y = canvas_height / 18.667
            font_size = int(min(text_x, text_y))
            canvas.itemconfig(text_1, font=("Inter", font_size * -1))
            rect_1_x1, rect_1_y1, rect_1_x2, rect_1_y2 = canvas.coords(rectangle_1)
            rect_1_center_x = (rect_1_x1 + rect_1_x2) / 2
            rect_1_center_y = (rect_1_y1 + rect_1_y2) / 2
            text_1_bbox = canvas.bbox(text_1)
            text_1_width = text_1_bbox[2] - text_1_bbox[0]
            text_1_height = text_1_bbox[3] - text_1_bbox[1]
            text_1_x = rect_1_center_x - text_1_width / 2
            text_1_y = rect_1_center_y - text_1_height / 2
            canvas.coords(text_1, text_1_x, text_1_y)
        if text_4:
            text_x = canvas_width / 18.667
            text_y = canvas_height / 18.667
            font_size = int(min(text_x, text_y))
            canvas.itemconfig(text_4, font=("Inter", font_size * -1))
            rect_1_x1, rect_1_y1, rect_1_x2, rect_1_y2 = canvas.coords(rectangle_4)
            rect_1_center_x = (rect_1_x1 + rect_1_x2) / 2
            rect_1_center_y = (rect_1_y1 + rect_1_y2) / 2
            text_1_bbox = canvas.bbox(text_4)
            text_1_width = text_1_bbox[2] - text_1_bbox[0]
            text_1_height = text_1_bbox[3] - text_1_bbox[1]
            text_1_x = rect_1_center_x - text_1_width / 2
            text_1_y = rect_1_center_y - text_1_height / 2
            canvas.coords(text_4, text_1_x, text_1_y)
        if text_5:
            text_x = canvas_width / 18.667
            text_y = canvas_height / 18.667
            font_size = int(min(text_x, text_y))
            canvas.itemconfig(text_5, font=("Inter", font_size * -1))
            rect_1_x1, rect_1_y1, rect_1_x2, rect_1_y2 = canvas.coords(rectangle_5)
            rect_1_center_x = (rect_1_x1 + rect_1_x2) / 2
            rect_1_center_y = (rect_1_y1 + rect_1_y2) / 2
            text_1_bbox = canvas.bbox(text_5)
            text_1_width = text_1_bbox[2] - text_1_bbox[0]
            text_1_height = text_1_bbox[3] - text_1_bbox[1]
            text_1_x = rect_1_center_x - text_1_width / 2
            text_1_y = rect_1_center_y - text_1_height / 2
            canvas.coords(text_5, text_1_x, text_1_y)

        # Images
        global camera_width, camera_height
        camera_width = (int)(canvas_width * 0.464)
        camera_height = (int)(canvas_height * 0.535)
        nonlocal video_0, video_1
        if video_0:
            video_0.change_dims(camera_width, camera_height)
        if video_1:
            video_1.change_dims(camera_width, camera_height)
        nonlocal rectangle_2, rectangle_3
        if rectangle_2:
            resized_image_2 = canvas.rectangle_image_2.resize((camera_width, camera_height))
            photo_image_2 = ImageTk.PhotoImage(resized_image_2)
            canvas.itemconfig(rectangle_2, image=photo_image_2)
            canvas.rectangle_2 = photo_image_2  # Keep a reference to prevent garbage collection
            canvas.coords(rectangle_2, canvas_width * 0.020, canvas_width * 0.429)
        if rectangle_3:
            resized_image_3 = canvas.rectangle_image_3.resize((camera_width, camera_height))
            photo_image_3 = ImageTk.PhotoImage(resized_image_3)
            canvas.itemconfig(rectangle_3, image=photo_image_3)
            canvas.rectangle_3 = photo_image_3  # Keep a reference to prevent garbage collection
            canvas.coords(rectangle_3, canvas_width * 0.517, canvas_width * 0.429)

        # Buttons
        nonlocal button_1, button_2, button_3
        if button_1:
            button_1_width = (int)(canvas_width * 0.157)
            button_1_height = (int)(canvas_height * 0.112)
            resized_button_image_1 = Image.open(relative_to_assets("image_1.png")).resize((button_1_width, button_1_height))
            button_image_1 = ImageTk.PhotoImage(resized_button_image_1)
            button_1.config(image=button_image_1)
            window.button_image_1 = button_image_1
        if button_2:
            button_2_width = (int)(canvas_width * 0.157)
            button_2_height = (int)(canvas_height * 0.112)
            resized_button_image_2 = Image.open(relative_to_assets("image_2.png")).resize((button_2_width, button_2_height))
            button_image_2 = ImageTk.PhotoImage(resized_button_image_2)
            button_2.config(image=button_image_2)
            window.button_image_2 = button_image_2
        if button_3:
            button_3_width = (int)(canvas_width * 0.157)
            button_3_height = (int)(canvas_height * 0.112)
            resized_button_image_3 = Image.open(relative_to_assets("image_3.png")).resize((button_3_width, button_3_height))
            button_image_3 = ImageTk.PhotoImage(resized_button_image_3)
            button_3.config(image=button_image_3)
            window.button_image_3 = button_image_3

    # Bind resizing events
    canvas.bind("<Configure>", resize_canvas)

    # Switch dot states to display "waiting" to the user
    global text_states
    text_states = {
        "text_generating": True
    }

    def update_waiting_text(text_item, initial_text, delay, id):
        dots = 0
        def update_text():
            if text_states[id]:
                nonlocal dots
                new_text = initial_text[:-3] + "." * (dots % 4)
                canvas.itemconfig(text_item, text=new_text)
                dots += 1
                if not False:
                    canvas.after(delay, update_text)

    update_waiting_text(text_1, "Generating Results...", 1000, "text_generating")

    # Button actions
    def push_yes_button():
        global confirm_status, click_button_event
        confirm_status = True
        click_button_event.set()

    def push_no_button():
        global confirm_status, click_button_event
        confirm_status = False
        click_button_event.set()
    
    def push_slider_button():
        print("Made it to slider button function!")
        global click_slider_button_event
        click_slider_button_event.set()

    def place_items_keep_video():
        # Present buttons to user
        nonlocal rectangle_1, text_1
        rectangle_1 = canvas.create_rectangle(
        54.0, 44.0, 354.0, 139.0,
        fill="#FFA629",
        outline="")
        text_1 = canvas.create_text(
        193.0,
        211.0,
        anchor="nw",
        text="Do you wish to proceed\nwith this swing?",
        fill="#000000",
        font=("Inter", 24 * -1))

        # Buttons
        nonlocal button_1, button_2
        window.button_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        button_1 = Button(
            window,
            image=window.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: push_yes_button(),
            relief="flat"
        )
        button_1.place(relx=0.75, rely=0.125, anchor="center")

        window.button_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        button_2 = Button(
            window,
            image=window.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: push_no_button(),
            relief="flat"
        )
        button_2.place(relx=0.75, rely=0.275, anchor="center")
        resize_canvas()
    
    def create_slider_1(frames_count, frames_0, frames_1):
        # First, hide the old objects
        nonlocal rectangle_1, text_1, button_1, button_2
        canvas.delete(rectangle_1)
        canvas.delete(text_1)
        button_1.destroy()
        button_2.destroy()
        rectangle_1 = None
        text_1 = None
        button_1 = None
        button_2 = None

        # Wrapper function for updating camera feeds
        def update_frame(frame_idx):
            global frame_idx_var
            frame_idx_var = frame_idx
            set_video_to_frame(frames_0, 0, frame_idx)
            set_video_to_frame(frames_1, 1, frame_idx)

        # Create a slider
        global canvas_width
        nonlocal slider
        slider = ttk.Scale(window, from_=0, to=frames_count - 1, value=0, orient=tk.HORIZONTAL, command=lambda value: update_frame(int(float(value))), length=(int)(canvas_width * 0.75))
        slider.place(relx=0.5, rely=0.35, anchor='center')

        # Info
        nonlocal rectangle_4, text_4
        rectangle_4 = canvas.create_rectangle(
        54.0, 44.0, 354.0, 139.0,
        fill="#FFA629",
        outline="")
        text_4 = canvas.create_text(
        193.0,
        211.0,
        anchor="nw",
        text="Drag the slider to\nthe STARTING frame.",
        fill="#000000",
        font=("Inter", 24 * -1))

        # Buttons
        nonlocal button_3
        window.button_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        button_3 = Button(
            window,
            image=window.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: push_slider_button(),
            relief="flat"
        )
        button_3.place(relx=0.75, rely=0.2, anchor="center")
        resize_canvas()
        update_frame(0)
        

    def create_slider_2(frames_count, frames_0, frames_1):

        def update_frames_again(frame_idx):
            global frame_idx_var
            frame_idx_var = frame_idx
            set_video_to_frame(frames_0, 0, frame_idx)
            set_video_to_frame(frames_1, 1, frame_idx)

        # Create a slider
        global canvas_width, frame_idx_var
        nonlocal slider
        slider.configure(from_=0, to=frames_count - 1, value=0, command=lambda value: update_frames_again(int(float(value))))
        slider.place(relx=0.5, rely=0.35, anchor='center')

        # Modify text
        canvas.itemconfig(text_4, text="Drag the slider to\nthe ENDING frame.")

    def create_generating_results():
        nonlocal rectangle_4, text_4, button_3, rectangle_2, rectangle_3, slider
        canvas.delete(rectangle_2)
        canvas.delete(rectangle_3)
        canvas.delete(rectangle_4)
        canvas.delete(text_4)
        slider.destroy()
        button_3.destroy()
        rectangle_2 = None
        rectangle_3 = None
        rectangle_4 = None
        button_3 = None
        text_4 = None

        # Generating Results empty box
        nonlocal rectangle_5, text_5
        rectangle_5 = canvas.create_rectangle(
        274.0, 208.0, 574.0, 240.0,
        fill="#FFA629",
        outline="")
        text_5 = canvas.create_text(
        293.0,
        211.0,
        anchor="nw",
        text="Generating Results...",
        fill="#000000",
        font=("Inter", 24 * -1))

        resize_canvas()
    
    def set_video_to_frame(frames, idx, frame_idx):
        if idx == 0:
            nonlocal video_0
            image, photo = video_0.get_image_and_photo(frames[frame_idx])
            canvas.itemconfig(rectangle_2, image=photo)
            canvas.rectangle_2 = photo
            canvas.rectangle_image_2 = image
            canvas.coords(rectangle_2, canvas_width * 0.020, canvas_height * 0.429)
        else:
            nonlocal video_1
            image, photo = video_1.get_image_and_photo(frames[frame_idx])
            canvas.itemconfig(rectangle_3, image=photo)
            canvas.rectangle_3 = photo
            canvas.rectangle_image_3 = image
            canvas.coords(rectangle_3, canvas_width * 0.517, canvas_height * 0.429)

    # Threading sequencing
    global click_button_event, click_slider_button_event
    click_button_event = threading.Event()
    click_slider_button_event = threading.Event()
    video_0 = None
    video_1 = None
    def thread_sequencing_initial():
        # Create video player objects
        global camera_width, camera_height
        nonlocal video_0, video_1
        video_0 = VideoPlayer(r"data_processing\video_data\output_1.mp4", canvas, window, rectangle_2, camera_width, camera_height, 0)
        video_1 = VideoPlayer(r"data_processing\video_data\output_2.mp4", canvas, window, rectangle_3, camera_width, camera_height, 1)
        min_time = min(video_0.get_clip_length(), video_1.get_clip_length())
        video_0.set_trim_length(min_time)
        video_1.set_trim_length(min_time)

        print("Created objects!")

        # Play videos on loop
        play_video_0_thread = threading.Thread(target = video_0.play_clip_on_loop)
        play_video_1_thread = threading.Thread(target = video_1.play_clip_on_loop)
        play_video_0_thread.start()
        play_video_1_thread.start()

        # Prompt the user to keep the videos
        window.after(0, place_items_keep_video)

        # Wait for button press event
        global confirm_status, click_button_event, terminate_early
        confirm_status = False
        if not terminate_early:
            click_button_event.wait()
        video_0.stop_playing()
        video_1.stop_playing()
        #click_button_event.clear()
        
        # No was pressed
        if not confirm_status:
            window.after(0, calibration_window)
            return

        # Get a collection of frames to display to the user
        frames_0 = video_0.get_collection_of_frames()
        frames_1 = video_1.get_collection_of_frames()
        frames_count = min(len(frames_0), len(frames_1))

        # Introduce slider here
        window.after(0, lambda: create_slider_1(frames_count, frames_0, frames_1))

        # Wait for first frame to be selected
        global click_slider_button_event
        if not terminate_early:
            click_slider_button_event.wait()
        click_slider_button_event.clear()
        
        # Wait for last frame to be selected
        print("Got first frame to be selected!")
        global frame_idx_var
        frames_0 = frames_0[frame_idx_var:]
        frames_1 = frames_1[frame_idx_var:]
        frames_count = min(len(frames_0), len(frames_1))

        window.after(0, lambda: create_slider_2(frames_count, frames_0, frames_1))
        if not terminate_early:
            click_slider_button_event.wait()
        click_slider_button_event.clear()

        # Update frames to new collection
        print("Got last frame to be selected!")
        frames_0 = frames_0[:frame_idx_var]
        frames_1 = frames_1[:frame_idx_var]

        # Now, display generating info...
        window.after(0, create_generating_results)

        # Also, create mp4 files
        video_0.convert_frames_to_video(r"data_processing\video_data\trimmed_out_0.mp4", frames_0)
        video_1.convert_frames_to_video(r"data_processing\video_data\trimmed_out_1.mp4", frames_1)
        
        # Now, process with AI!
        execute_process_video()

        # And finally, proceed to next window!
        window.after(0, next_window)

    def calibration_window():
        from gui_module.build import gui_calibration
        close_window(window, width, height, x, y)
        gui_calibration.main()

    def next_window():
        #from gui_module.build import gui_results_summary
        #close_window(window, width, height, x, y)
        #gui_results_summary.main()
        print("Ended gui_generating_results.py!")

    # Start the video playback as a thread
    initial_data_thread = threading.Thread(target = thread_sequencing_initial)
    initial_data_thread.start()

    window.mainloop()


if __name__ == "__main__":
    main()