from pathlib import Path
from gui_module.build import gui_calibration
from gui_module.build import gui_help
from tkinter import Tk, Canvas, Button, PhotoImage
from PIL import Image, ImageTk
import json
import os
import threading

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r".\assets\frame11")

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

def close_window(window, width, height, x, y, forceShutdown = False):
    if window.attributes("-fullscreen") == 0:
            x, y = get_window_position(window)
    save_window_state(width, height, window.attributes("-fullscreen"), x, y, window.state() == 'zoomed')
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
        bg="#FFFFFF",
        height=448,
        width=797,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas.pack(fill="both", expand=True)

    # Add custom attributes to the canvas object
    canvas.image_1 = None
    canvas.image_2 = None

    # Resize background image
    def resize_background(event=None):
        # Get the size of the canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

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
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Button 1
        button_1_width = (int)(canvas_width * 0.31)
        button_1_height = (int)(canvas_height * 0.178)
        resized_button_image_1 = Image.open(relative_to_assets("button_1.png")).resize((button_1_width, button_1_height))
        button_image_1 = ImageTk.PhotoImage(resized_button_image_1)
        button_1.config(image=button_image_1)
        button_1.image = button_image_1

        # Button 2
        button_2_width = (int)(canvas_width * 0.12)
        button_2_height = (int)(canvas_height * 0.071)
        resized_button_image_2 = Image.open(relative_to_assets("button_2.png")).resize((button_2_width, button_2_height))
        button_image_2 = ImageTk.PhotoImage(resized_button_image_2)
        button_2.config(image=button_image_2)
        button_2.image = button_image_2

        # Resize the background image
        resize_background()

        # AutoCaddie title
        image_2_width = (int)(canvas_width * 0.44)
        image_2_height = (int)(canvas_height * 0.18)
        resized_image_2 = image_image_2.resize((image_2_width, image_2_height))
        photo_image_2 = ImageTk.PhotoImage(resized_image_2)
        canvas.itemconfig(image_2, image=photo_image_2)
        canvas.image_2 = photo_image_2  # Keep a reference to prevent garbage collection
        canvas.coords(image_2, canvas_width / 2, canvas_height * 0.1)

    # Bind resizing events
    canvas.bind("<Configure>", resize_canvas)

    # Load the background image
    global image_image_1, image_image_2, image_1, image_2
    image_image_1 = Image.open(relative_to_assets("image_1.png"))
    photo_image_1 = ImageTk.PhotoImage(image_image_1)
    image_1 = canvas.create_image(0, 0, anchor="nw", image=photo_image_1)

    # Start button
    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: press_start_button(),
        relief="flat"
    )
    button_1.place(relx=0.5, rely=0.725, anchor="center")

    # Start button action
    def press_start_button():
        print("Debug: Start button clicked")
        # Open the window
        close_window(window, width, height, x, y)
        gui_calibration.main()

    # Help button
    button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
    button_2 = Button(
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: press_help_button(),
        relief="flat"
    )
    button_2.place(relx=0.5, rely=0.9, anchor="center")

    # Help button action
    def press_help_button():
        print("Debug: Help button clicked")
        # Open the window
        close_window(window, width, height, x, y)
        gui_help.main()

    # AutoCaddie title
    image_image_2 = Image.open(relative_to_assets("image_2.png"))
    photo_image_2 = ImageTk.PhotoImage(image_image_2)
    image_2 = canvas.create_image(0, 0, anchor="center", image=photo_image_2)

    # Run the window
    window.mainloop()

if __name__ == "__main__":
    main()
