
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
import json
import threading
import time
from data_processing.serial_bluetooth_communication import stop_bluetooth, receive_packet_async
from algorithms.cv.convert_video_to_display import close_cameras
from data_processing.record_imu import receive_imu_data
from PIL import Image, ImageTk
from data_processing.camera_holder import CameraHolder
import os

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, ttk, Canvas, Entry, Text, Button, PhotoImage
import tkinter as tk


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r".\assets\frame3")

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
    window.protocol("WM_DELETE_WINDOW", lambda: close_window(window, width, height, x, y, True, True))
    
    return window

def close_window(window, width, height, x, y, close, forceShutdown = False):
    if window.attributes("-fullscreen") == 0:
            x, y = get_window_position(window)
    save_window_state(width, height, window.attributes("-fullscreen"), x, y, window.state() == 'zoomed')
    global terminate
    terminate = True

    if close:
        global ser_out, cameras
        stop_bluetooth(ser_out)
        close_cameras(cameras)
    window.destroy()

    if forceShutdown:
        os._exit(1)

def get_window_position(window):
    geometry_string = window.geometry()
    x, y = map(int, geometry_string.split('+')[1:])
    return x, y

def window_event(window):
    if window.state() != 'zoomed':
        return window.winfo_width(), window.winfo_height()
    else:
        return 0, 0

def main(camera_holder_1 = None, camera_holder_2 = None, camera_holder_3 = None, camera_holder_4 = None, ser_out_ref = None, cameras_ref = None):
    global ser_out, cameras, video_1, video_2, video_3, video_4
    ser_out = ser_out_ref
    cameras = cameras_ref
    video_1 = camera_holder_1
    video_2 = camera_holder_2
    video_3 = camera_holder_3
    video_4 = camera_holder_4
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
    global terminate, frame_1, photo_1, cur_cam_idx, pause
    terminate = False
    frame_1 = None
    photo_1 = None
    cur_cam_idx = 0
    pause = False

    # Objects
    image_image_1 = Image.open(relative_to_assets("image_1.png"))
    photo_image_1 = ImageTk.PhotoImage(image_image_1)
    image_1 = canvas.create_image(0, 0, anchor="nw", image=photo_image_1)

    # Temp rectangle
    dark_black_image = Image.new("RGB", (370, 240), "#1E1E1E")
    photo_image_dark_black = ImageTk.PhotoImage(dark_black_image)

    # Set rectangle 1
    rectangle_1 = None
    canvas.rectangle_1 = None
    canvas.rectangle_image_1 = None
    rectangle_1_index = 0
    def set_rectangle_1():
        nonlocal rectangle_1, rectangle_1_index
        if rectangle_1_index == 0:
            if os.path.exists(r"data_processing\imu_data\quaternion_graph_1.png"):
                graph_image = Image.open(r"data_processing\imu_data\quaternion_graph_1.png")
                photo_graph_image = ImageTk.PhotoImage(graph_image)
                rectangle_1 = canvas.create_image(411, 192, anchor="nw", image=photo_graph_image)
                canvas.rectangle_1 = photo_graph_image
                canvas.rectangle_image_1 = graph_image
            else:
                rectangle_1 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
                canvas.rectangle_1 = photo_image_dark_black
                canvas.rectangle_image_1 = dark_black_image
        elif rectangle_1_index == 1:
            if os.path.exists(r"data_processing\imu_data\roll_graph_1.png"):
                graph_image = Image.open(r"data_processing\imu_data\roll_graph_1.png")
                photo_graph_image = ImageTk.PhotoImage(graph_image)
                rectangle_1 = canvas.create_image(411, 192, anchor="nw", image=photo_graph_image)
                canvas.rectangle_1 = photo_graph_image
                canvas.rectangle_image_1 = graph_image
            else:
                rectangle_1 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
                canvas.rectangle_1 = photo_image_dark_black
                canvas.rectangle_image_1 = dark_black_image
        elif rectangle_1_index == 2:
            if os.path.exists(r"data_processing\imu_data\pitch_graph_1.png"):
                graph_image = Image.open(r"data_processing\imu_data\pitch_graph_1.png")
                photo_graph_image = ImageTk.PhotoImage(graph_image)
                rectangle_1 = canvas.create_image(411, 192, anchor="nw", image=photo_graph_image)
                canvas.rectangle_1 = photo_graph_image
                canvas.rectangle_image_1 = graph_image
            else:
                rectangle_1 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
                canvas.rectangle_1 = photo_image_dark_black
                canvas.rectangle_image_1 = dark_black_image
        else:
            if os.path.exists(r"data_processing\imu_data\yaw_graph_1.png"):
                graph_image = Image.open(r"data_processing\imu_data\yaw_graph_1.png")
                photo_graph_image = ImageTk.PhotoImage(graph_image)
                rectangle_1 = canvas.create_image(411, 192, anchor="nw", image=photo_graph_image)
                canvas.rectangle_1 = photo_graph_image
                canvas.rectangle_image_1 = graph_image
            else:
                rectangle_1 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
                canvas.rectangle_1 = photo_image_dark_black
                canvas.rectangle_image_1 = dark_black_image

    # Set rectangle 2
    rectangle_2 = None
    canvas.rectangle_2 = None
    canvas.rectangle_image_2 = None
    rectangle_2_index = 0
    def set_rectangle_2():
        nonlocal rectangle_2, rectangle_2_index
        if rectangle_2_index == 0:
            if os.path.exists(r"data_processing\imu_data\quaternion_graph_2.png"):
                graph_image = Image.open(r"data_processing\imu_data\quaternion_graph_2.png")
                photo_graph_image = ImageTk.PhotoImage(graph_image)
                rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_graph_image)
                canvas.rectangle_2 = photo_graph_image
                canvas.rectangle_image_2 = graph_image
            else:
                rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
                canvas.rectangle_2 = photo_image_dark_black
                canvas.rectangle_image_2 = dark_black_image
        elif rectangle_2_index == 1:
            if os.path.exists(r"data_processing\imu_data\roll_graph_2.png"):
                graph_image = Image.open(r"data_processing\imu_data\roll_graph_2.png")
                photo_graph_image = ImageTk.PhotoImage(graph_image)
                rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_graph_image)
                canvas.rectangle_2 = photo_graph_image
                canvas.rectangle_image_2 = graph_image
            else:
                rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
                canvas.rectangle_2 = photo_image_dark_black
                canvas.rectangle_image_2 = dark_black_image
        elif rectangle_2_index == 2:
            if os.path.exists(r"data_processing\imu_data\pitch_graph_2.png"):
                graph_image = Image.open(r"data_processing\imu_data\pitch_graph_2.png")
                photo_graph_image = ImageTk.PhotoImage(graph_image)
                rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_graph_image)
                canvas.rectangle_2 = photo_graph_image
                canvas.rectangle_image_2 = graph_image
            else:
                rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
                canvas.rectangle_2 = photo_image_dark_black
                canvas.rectangle_image_2 = dark_black_image
        else:
            if os.path.exists(r"data_processing\imu_data\yaw_graph_2.png"):
                graph_image = Image.open(r"data_processing\imu_data\yaw_graph_2.png")
                photo_graph_image = ImageTk.PhotoImage(graph_image)
                rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_graph_image)
                canvas.rectangle_2 = photo_graph_image
                canvas.rectangle_image_2 = graph_image
            else:
                rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)
                canvas.rectangle_2 = photo_image_dark_black
                canvas.rectangle_image_2 = dark_black_image

    set_rectangle_1()
    set_rectangle_2()

    # Load reference RPY image
    reference_image = Image.open(relative_to_assets("reference_rpy.png"))
    photo_reference_image = ImageTk.PhotoImage(reference_image)
    reference_image_obj = canvas.create_image(canvas_width * 0.5, canvas_height * 0.28, anchor="center", image=photo_reference_image)
    canvas.reference_image = photo_reference_image

    # White text rectangle
    rectangle_3 = canvas.create_rectangle(
        229.0,
        48.0,
        569.0,
        113.0,
        fill="#FFFFFF",
        outline="")

    # Metric Label
    text_2 = canvas.create_text(
        273.0,
        69.0,
        anchor="nw",
        text="INITIAL POSTURE",
        fill="#000000",
        font=("Inter", 24 * -1),
        justify="center"
    )

    # AutoCaddie title
    text_1 = canvas.create_text(
        16.0,
        12.0,
        anchor="nw",
        text="AutoCaddie",
        fill="#1E1E1E",
        font=("Inter SemiBold", 24 * -1),
        justify="center"
    )

    # Left button
    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: click_left_button(),
        relief="flat"
    )
    #button_1.place(
    #    x=48.0,
    #    y=48.0,
    #    width=92.0,
    #    height=72.0
    #)
    button_1.place(relx = 0.118, rely = 0.187, anchor="center")

    # Right button
    button_image_2 = PhotoImage(
        file=relative_to_assets("button_2.png"))
    button_2 = Button(
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: click_right_button(),
        relief="flat"
    )
    #button_2.place(
    #    x=657.0,
    #    y=48.0,
    #    width=92.0,
    #    height=72.0
    #)
    button_2.place(relx = 0.882, rely = 0.187, anchor="center")

    # Resize background image
    def resize_background(event=None):

        # Resize the image to fit the canvas size
        resized_image_1 = image_image_1.resize((canvas_width, canvas_height))

        # Convert the resized image to a Tkinter-compatible format
        photo_image_1 = ImageTk.PhotoImage(resized_image_1)

        # Update the canvas with the resized image
        canvas.itemconfig(image_1, image=photo_image_1)
        canvas.image_1 = photo_image_1  # Keep a reference to prevent garbage collection

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
        new_cords = [canvas_width * 0.287, canvas_height * 0.03, canvas_width * 0.713, canvas_height * 0.15]
        canvas.coords(rectangle_3, *new_cords)

        # Text
        text_x = canvas_width / 18.667
        text_y = canvas_height / 18.667
        font_size = int(min(text_x, text_y))

        canvas.itemconfig(text_1, font=("Inter SemiBold", font_size * -1))
        canvas.coords(text_1, canvas_width * 0.02, canvas_height * 0.027)

        canvas.itemconfig(text_2, font=("Inter", font_size * -1))
        rect_x1, rect_y1, rect_x2, rect_y2 = canvas.coords(rectangle_3)
        rect_center_x = (rect_x1 + rect_x2) / 2
        rect_center_y = (rect_y1 + rect_y2) / 2
        text_2_bbox = canvas.bbox(text_2)
        text_2_width = text_2_bbox[2] - text_2_bbox[0]
        text_2_height = text_2_bbox[3] - text_2_bbox[1]
        text_2_x = rect_center_x - text_2_width / 2
        text_2_y = rect_center_y - text_2_height / 2
        canvas.coords(text_2, text_2_x, text_2_y)
                
        # Images
        global camera_width, camera_height
        camera_width = (int)(canvas_width * 0.464)
        camera_height = (int)(canvas_height * 0.535)

        # Rectangles!
        canvas.coords(rectangle_1, canvas_width * 0.020, canvas_height * 0.429)
        canvas.coords(rectangle_2, canvas_width * 0.516, canvas_height * 0.429)
        resized_rectangle = canvas.rectangle_image_2.resize((camera_width, camera_height))
        photo_rectangle_image = ImageTk.PhotoImage(resized_rectangle)
        canvas.itemconfig(rectangle_2, image = photo_rectangle_image)
        canvas.rectangle_2 = photo_rectangle_image
        resized_rectangle = canvas.rectangle_image_1.resize((camera_width, camera_height))
        photo_rectangle_image = ImageTk.PhotoImage(resized_rectangle)
        canvas.itemconfig(rectangle_1, image = photo_rectangle_image)
        canvas.rectangle_1 = photo_rectangle_image

        # Button 1
        button_1_width = (int)(canvas_width * 0.115)
        button_1_height = (int)(canvas_height * 0.161)
        resized_button_image_1 = Image.open(relative_to_assets("button_1.png")).resize((button_1_width, button_1_height))
        button_image_1 = ImageTk.PhotoImage(resized_button_image_1)
        button_1.config(image=button_image_1)
        button_1.image = button_image_1
        
        # Button 2
        button_2_width = (int)(canvas_width * 0.115)
        button_2_height = (int)(canvas_height * 0.161)
        resized_button_image_2 = Image.open(relative_to_assets("button_2.png")).resize((button_2_width, button_2_height))
        button_image_2 = ImageTk.PhotoImage(resized_button_image_2)
        button_2.config(image=button_image_2)
        button_2.image = button_image_2

        # Reference image
        reference_image = Image.open(relative_to_assets("reference_rpy.png")).resize((int(canvas_width * 0.3), int(canvas_height * 0.2)))
        photo_reference_image = ImageTk.PhotoImage(reference_image)
        canvas.itemconfig(reference_image_obj, image=photo_reference_image)
        canvas.reference_image = photo_reference_image
        canvas.coords(reference_image_obj, canvas_width * 0.5, canvas_height * 0.28)
    
    # Bind resizing events
    canvas.bind("<Configure>", resize_canvas)

    def switch_quat_1(event):
        nonlocal rectangle_1_index
        rectangle_1_index += 1
        if rectangle_1_index == 4:
            rectangle_1_index = 0
        canvas.tag_unbind(rectangle_1, "<Button-1>")
        set_rectangle_1()
        resize_canvas()
        canvas.tag_bind(rectangle_1, "<Button-1>", switch_quat_1)

    def switch_quat_2(event):
        nonlocal rectangle_2_index
        rectangle_2_index += 1
        if rectangle_2_index == 4:
            rectangle_2_index = 0
        canvas.tag_unbind(rectangle_2, "<Button-1>")
        set_rectangle_2()
        resize_canvas()
        canvas.tag_bind(rectangle_2, "<Button-1>", switch_quat_2)

    # Bind click events to the rectangles
    canvas.tag_bind(rectangle_1, "<Button-1>", switch_quat_1)
    canvas.tag_bind(rectangle_2, "<Button-1>", switch_quat_2)

    def wait_for_ser_out():
        global terminate, ser_out
        while terminate is False:
            line = receive_packet_async(ser_out)
            
            # Idle
            if line == None:
                continue

            # We got liftoff!
            if line == "READY":
                window.after(0, click_recording_button)
    
    # Switch to the recording gui
    def click_recording_button():
        global ser_out, cameras

        # Start data recording process
        data_thread = threading.Thread(target = receive_imu_data, args=(ser_out,))
        data_thread.start()

        # Next window
        from gui_module.build import gui_recording
        close_window(window, width, height, x, y, False)
        gui_recording.main(ser_out, cameras[0], cameras[1], data_thread)
        
    def click_left_button():
        global video_1, video_2, video_3, video_4, ser_out, cameras
        from gui_module.build import gui_results_main
        close_window(window, width, height, x, y, False)
        gui_results_main.main(video_1, video_2, video_3, video_4, ser_out, cameras)

    def click_right_button():
        global video_1, video_2, video_3, video_4, ser_out, cameras
        from gui_module.build import gui_results_arm
        close_window(window, width, height, x, y, False)
        gui_results_arm.main(video_1, video_2, video_3, video_4, ser_out, cameras)

    # Start threads
    listen_for_button = threading.Thread(target = wait_for_ser_out)
    listen_for_button.start()
    window.mainloop()
