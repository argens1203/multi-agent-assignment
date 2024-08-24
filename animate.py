from tkinter import *
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")


import random
import time


class RegrMagic(object):
    """Mock for function Regr_magic()"""

    def __init__(self):
        self.x = 0

    def __call__(self):
        time.sleep(0.01)
        self.x += 1
        return self.x, random.random()


regr_magic = RegrMagic()


def frames():
    while True:
        yield regr_magic()


class Application(Frame):
    """A GUI app with some buttons."""

    def __init__(self, master):
        """Initialze frame"""
        Frame.__init__(self, master)
        self.grid()
        self.master = master
        self.anim_fig, self.anim_ax = plt.subplots()
        self.start_anim()

    def start_anim(self):

        self.canvas = FigureCanvasTkAgg(self.anim_fig, self.master)
        self.canvas.get_tk_widget().grid()

        # toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        # toolbar.update()
        # self.canvas._tkcanvas.pack()

        self.ani = animation.FuncAnimation(
            self.anim_fig, self.animate, frames=frames, interval=100, save_count=100
        )

    xs = []
    ys = []

    def animate(self, args):
        x, y = args
        self.xs.append(x)
        self.ys.append(y)
        self.anim_ax.clear()
        self.anim_ax.plot(self.xs, self.ys)

        self.anim_fig.canvas.mpl_connect("close_event", self.on_close)

    def on_close(self, e):
        self.master.quit()


# Building the window
root = Tk()

app = Application(root)

# MainLoop
root.mainloop()
