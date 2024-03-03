from pathlib import Path
import gui_calibration
import gui_help
from tkinter import Tk, Canvas, Button, PhotoImage
from PIL import Image, ImageTk
import json

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r".\assets\frame11")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def save_window_state(width, height, fullscreen):
    with open("assets\window_state.json", "w") as f:
        json.dump({"width": width, "height": height, "fullscreen": fullscreen}, f)

def load_window_state():
    try:
        with open("assets\window_state.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    
def create_window(width, height, fullscreen):
    window = Tk()
    window.geometry(f"{width}x{height}")
    window.configure(bg="#FFFFFF")
    if fullscreen:
        window.attributes("-fullscreen", True)
    
    # Save window state before closing
    window.protocol("WM_DELETE_WINDOW", lambda: close_window(window))

    center_window(window)
    
    return window

def close_window(window):
    save_window_state(window.winfo_width(), window.winfo_height(), window.attributes("-fullscreen"))
    window.destroy()

def center_window(window):
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    window.geometry("+{}+{}".format(x, y))

def main():
    saved_state = load_window_state()
    if saved_state:
        width, height, fullscreen = saved_state["width"], saved_state["height"], saved_state["fullscreen"]
    else:
        width, height, fullscreen = 797, 448, False
    
    window = create_window(width, height, fullscreen)

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
        window.destroy()
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
        window.destroy()
        gui_help.main()

    # AutoCaddie title
    image_image_2 = Image.open(relative_to_assets("image_2.png"))
    photo_image_2 = ImageTk.PhotoImage(image_image_2)
    image_2 = canvas.create_image(0, 0, anchor="center", image=photo_image_2)

    # Run the window
    window.mainloop()

if __name__ == "__main__":
    main()
