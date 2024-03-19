
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
from gui_module.build import gui_recording
import json
import threading
from PIL import Image, ImageTk
from data_processing.serial_bluetooth_communication import start_bluetooth, receive_packet, stop_bluetooth
from data_processing.record_imu import receive_imu_data
from data_processing.find_serial_port import find_com
from algorithms.cv.find_valid_camera_indices import find_camera_indices
from algorithms.cv.convert_video_to_display import open_cameras, close_cameras, get_frame_from_camera
import cv2

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r".\assets\frame9")

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
    window.protocol("WM_DELETE_WINDOW", lambda: close_window(window, width, height, x, y, True))
    
    return window

def close_window(window, width, height, x, y, closeBluetooth):
    if window.attributes("-fullscreen") == 0:
            x, y = get_window_position(window)
    save_window_state(width, height, window.attributes("-fullscreen"), x, y, window.state() == 'zoomed')
    global cameras
    if cameras is not None:
        close_cameras(cameras)
    
    if closeBluetooth:
        global terminate_bluetooth
        terminate_bluetooth = True
        stop_bluetooth(ser_out)
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

def main():
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

    # Calibration states
    global connected_bluetooth, connected_cameras, cameras
    connected_bluetooth = False
    connected_cameras = 0
    cameras = None
            
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
        new_cords = [canvas_width * 0.020, canvas_height * 0.107, canvas_width * 0.484, canvas_height * 0.393]
        canvas.coords(rectangle_3, *new_cords)
        new_cords = [canvas_width * 0.516, canvas_height * 0.107, canvas_width * 0.98, canvas_height * 0.393]
        canvas.coords(rectangle_4, *new_cords)

        # Images
        global camera_width, camera_height
        camera_width = (int)(canvas_width * 0.464)
        camera_height = (int)(canvas_height * 0.535)
        # Cameras have not been found yet...
        if connected_cameras < 2 :
            resized_rectangle = dark_black_image.resize((camera_width, camera_height))
            photo_image_dark_black = ImageTk.PhotoImage(resized_rectangle)
            # Rectangle 2
            canvas.itemconfig(rectangle_2, image=photo_image_dark_black)
            canvas.rectangle_2 = photo_image_dark_black
            canvas.coords(rectangle_2, canvas_width * 0.517, canvas_height * 0.429)
            # Rectangle 1
            if connected_cameras == 0:          # THIS IS MODIFIED FOR TESTING!!!
                canvas.itemconfig(rectangle_1, image=photo_image_dark_black)
                canvas.rectangle_1 = photo_image_dark_black
                canvas.coords(rectangle_1, canvas_width * 0.020, canvas_height * 0.429)

        # Text
        text_x = canvas_width / 18.667
        text_y = canvas_height / 18.667
        font_size = int(min(text_x, text_y))
        canvas.itemconfig(text_1, font=("Inter", font_size * -1))
        canvas.coords(text_1, canvas_width * 0.683, canvas_height * 0.630)
        canvas.itemconfig(text_2, font=("Inter", font_size * -1))
        canvas.coords(text_2, canvas_width * 0.179, canvas_height * 0.183)
        canvas.itemconfig(text_3, font=("Inter", font_size * -1))
        canvas.coords(text_3, canvas_width * 0.683, canvas_height * 0.183)
        canvas.itemconfig(text_4, font=("Inter", font_size * -1))
        canvas.coords(text_4, canvas_width * 0.197, canvas_height * 0.630)
        canvas.itemconfig(text_5, font=("Inter", font_size * -1))
        canvas.coords(text_5, canvas_width * 0.02, canvas_height * 0.027)

    # Bind resizing events
    canvas.bind("<Configure>", resize_canvas)

    # Image 1
    image_image_1 = Image.open(relative_to_assets("image_1.png"))
    photo_image_1 = ImageTk.PhotoImage(image_image_1)
    image_1 = canvas.create_image(0, 0, anchor="nw", image=photo_image_1)

    # Rectangle 1 (REPLACE WITH IMAGE)
    #rectangle_1 = canvas.create_rectangle(
    #    16.0,
    #    192.0,
    #    386.0,
    #    432.0,
    #    fill="#1E1E1E",
    #    outline="")
    dark_black_image = Image.new("RGB", (370, 240), "#1E1E1E")
    photo_image_dark_black = ImageTk.PhotoImage(dark_black_image)
    rectangle_1 = canvas.create_image(16, 192, anchor="nw", image=photo_image_dark_black)

    # Rectangle 2 (REPLACE WITH IMAGE)
    #rectangle_2 = canvas.create_rectangle(
    #    411.0,
    #    192.0,
    #    781.0,
    #    432.0,
    #    fill="#1E1E1E",
    #    outline="")
    rectangle_2 = canvas.create_image(411, 192, anchor="nw", image=photo_image_dark_black)

    # Text 1
    text_1 = canvas.create_text(
        544.0,
        282.0,
        anchor="nw",
        text="VIDEO 2\nWaiting...",
        fill="#FFFFFF",
        font=("Inter", 24 * -1)
    )

    # Rectangle 3
    rectangle_3 = canvas.create_rectangle(
        16.26530647277832,
        48.0,
        386.2653064727783,
        176.0,
        fill="#FF0000",
        outline="")

    # Rectangle 4
    rectangle_4 = canvas.create_rectangle(
        411.0,
        48.0,
        781.0,
        176.0,
        fill="#FFFFFF",
        outline="")

    # Text 2
    text_2 = canvas.create_text(
        143.0,
        82.0,
        anchor="nw",
        text="STATUS\nWaiting...",
        fill="#000000",
        font=("Inter", 24 * -1)
    )

    # Text 3
    text_3 = canvas.create_text(
        544.0,
        82.0,
        anchor="nw",
        text="MCU\nWaiting...",
        fill="#000000",
        font=("Inter", 24 * -1)
    )

    # Text 4
    text_4 = canvas.create_text(
        143.0,
        282.0,
        anchor="nw",
        text="VIDEO 1\nWaiting...",
        fill="#FFFFFF",
        font=("Inter", 24 * -1)
    )

    # Text 5
    text_5 = canvas.create_text(
        16.0,
        12.0,
        anchor="nw",
        text="AutoCaddie",
        fill="#1E1E1E",
        font=("Inter SemiBold", 24 * -1)
    )

    # Bluetooth container
    def bluetooth_serial():
        print("Entered bluetooth serial")

        # Ensure bluetooth initialized correctly
        global ser_out, terminate_bluetooth, connected_bluetooth
        ser_out = None
        terminate_bluetooth = False

        # Find the correct out port
        com = find_com()
        while com is None and terminate_bluetooth is False:
            print("Finding Bluetooth Com")
            com = find_com()

        # Establish a bluetooth connection
        baud = 9600
        ser_out = start_bluetooth(com, baud)
        while ser_out is None and terminate_bluetooth is False:
            print("Starting Bluetooth")
            ser_out = start_bluetooth(com, baud)

        # Continuously poll for start packet
        while connected_bluetooth is False and terminate_bluetooth is False:

            # Poll infinitely
            line = receive_packet(ser_out)
            if line == None:
                continue

            # Start signal?
            if line == "READY":
                connected_bluetooth = True

        # Continuously poll for "Start Recording" signal
        while terminate_bluetooth is False:
            
            # Poll infinitely
            line = receive_packet(ser_out)
            if line == None:
                continue

            # Button pressed on hardware?
            if line == "START":

                # Good to go?
                if connected_cameras == 2:
                    terminate_bluetooth = True
                    next_window()

                # Not good to go
                else:
                    print("Debug: Button was pressed, cameras not ready.")

    # Camera initializing method
    def start_cameras():
        print("Entered Start Cameras")

        # Get valid camera indices
        global terminate_bluetooth, connected_cameras
        camera_indices = find_camera_indices()
        while len(camera_indices) < 1:                  # THIS IS MODIFIED FOR TESTING!!!
            print("Will retry looking for cameras")
            camera_indices = find_camera_indices()
            if terminate_bluetooth == True:
                return
        global cameras
        cameras = open_cameras(camera_indices)
        connected_cameras = len(camera_indices)
        print(f"Debug: Found {connected_cameras} cameras")
        if not terminate_bluetooth:
            window.after(1, update_camera_feeds)  # 1 milisecond

        # Select the first 2 camera feeds in the list
        #video_stream_1 = capture_camera(camera_indices[0])
        #video_stream_2 = capture_camera(camera_indices[1])

        # Replace the rectangles with images
    
    def update_camera_feeds():
        # This code is in the main thread...
        #current_thread = threading.current_thread()
        #if current_thread.name == "MainThread":
        #    print("This code is running in the main thread.")
        #else:
        #    print(f"This code is running in a separate thread named '{current_thread.name}'.")
        global terminate_bluetooth, cameras
        global camera_width, camera_height
        captured_frame = get_frame_from_camera(cameras[0], camera_width, camera_height)
        canvas.itemconfig(rectangle_1, image=captured_frame)
        canvas.rectangle_1 = captured_frame  # Prevent garbage collection
        canvas.coords(rectangle_1, canvas_width * 0.020, canvas_height * 0.429)
        if connected_cameras > 1:               # THIS IS MODIFIED FOR TESTING!!!
            captured_frame = get_frame_from_camera(cameras[1], camera_width, camera_height)
            canvas.itemconfig(rectangle_2, image=captured_frame)
            canvas.rectangle_2 = captured_frame  # Prevent garbage collection
            canvas.coords(rectangle_2, canvas_width * 0.517, canvas_height * 0.429)

        #update_video(canvas, cameras[0], rectangle_1, canvas.rectangle_1, camera_width, camera_height, canvas_width * 0.020, canvas_height * 0.429)
        #update_video(canvas, cameras[1], rectangle_2, canvas.rectangle_2, camera_width, camera_height, canvas_width * 0.517, canvas_height * 0.429)
        if not terminate_bluetooth:
            window.after(100, update_camera_feeds)  # 1 milisecond

    # Move to the recording frame
    def next_window():
        print("Debug: Calibration Succeeded.")

        # Start data recording process
        data_thread = threading.Thread(target = receive_imu_data, args=(ser_out,))
        data_thread.start()

        # Open the window
        close_window(window, width, height, x, y, False)
        gui_recording.main(ser_out)

    # Start the camera finding on a separate thread
    camera_thread = threading.Thread(target = start_cameras)
    camera_thread.start()

    # Start the Bluetooth connection on a separate thread
    bluetooth_thread = threading.Thread(target = bluetooth_serial)
    bluetooth_thread.start()

    window.mainloop()

if __name__ == "__main__":
    main()