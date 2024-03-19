
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r".\assets\frame7")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("797x448")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 448,
    width = 797,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    398.0,
    224.0,
    image=image_image_1
)

canvas.create_rectangle(
    274.0,
    208.0,
    524.0,
    240.0,
    fill="#FFA629",
    outline="")

canvas.create_text(
    293.0,
    211.0,
    anchor="nw",
    text="Generating Results",
    fill="#000000",
    font=("Inter", 24 * -1)
)
window.resizable(False, False)
window.mainloop()