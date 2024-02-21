from gui_module.autocaddie_gui import AutocaddieFrame
import wx

if __name__ == "__main__":
    app = wx.App(False)
    frame = AutocaddieFrame()
    frame.Show()
    app.MainLoop()
