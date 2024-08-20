import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

matplotlib.use("TkAgg")

import random
import threading


class Sim:
    def __init__(self, world):
        self.world = world
        self.speed = 1.0
        self.stop = False

    def stopAnim(self, d):
        self.stop = True

    def startAnim(self, d):
        self.start = True
        self.stop = False
        self.foo()

    def foo(self):
        self.advance(None)
        if not (self.stop):
            self.timer = threading.Timer(self.speed, self.foo)
            self.timer.start()

    def advance(self, event):
        self.time += 1
        self.world.step()
        self.mat.set_data(self.world.get_repr())
        plt.title("t = " + str(self.time))
        plt.show()

    def initAnim(self, event):
        self.time = 0

        self.world.init()

        self.mat = self.ax.matshow(self.world.get_repr(), cmap=plt.cm.seismic)
        self.mat.set_data(self.world.get_repr())
        plt.title("t = " + str(self.time))
        plt.show()

    def updateSpeed(self, val: float | None):
        if val:
            self.speed = 1 / val

    def run(self):
        random.seed()
        fig, self.ax = plt.subplots()
        self.ax.axis("off")
        plt.title("Shelling's Segregation Model")

        self.init_slider()
        self.init_buttons()

        self.initAnim(None)
        self.updateSpeed(None)

        return self.world.get_metrics()

    def init_slider(self):
        axspeed = plt.axes([0.175, 0.05, 0.65, 0.03])
        self.sspeed = Slider(axspeed, "Speed", 0.1, 10.0, valinit=1.0)
        self.sspeed.on_changed(self.updateSpeed)

    def init_buttons(self):
        axnext = plt.axes([0.85, 0.15, 0.1, 0.075])
        axstart = plt.axes([0.85, 0.25, 0.1, 0.075])
        axstop = plt.axes([0.85, 0.35, 0.1, 0.075])
        axinit = plt.axes([0.85, 0.45, 0.1, 0.075])

        self.bnext = Button(axnext, "Next")
        self.bnext.on_clicked(self.advance)
        self.bstart = Button(axstart, "Start")
        self.bstart.on_clicked(self.startAnim)
        self.bstop = Button(axstop, "Stop")
        self.bstop.on_clicked(self.stopAnim)
        self.binit = Button(axinit, "Init")
        self.binit.on_clicked(self.initAnim)
