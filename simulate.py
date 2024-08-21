import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

matplotlib.use("TkAgg")

import random
import threading
import math
import numpy


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
        self.show_data()
        plt.title("t = " + str(self.time))
        plt.show()

    def initAnim(self, event):
        self.time = 0

        self.world.init()

        self.create_lookup_for_world()
        transformed = self.get_transformed_data()
        self.mat = self.ax.matshow(transformed, cmap=plt.cm.seismic)

        self.show_data()
        plt.title("t = " + str(self.time))
        plt.show()

    def create_lookup_for_world(self):
        s = set()
        minimum = math.inf
        maximum = -math.inf
        for row in self.world.get_repr():
            for cell in row:
                if type(cell) is int or type(cell) is float:
                    minimum = min(minimum, cell)
                    maximum = max(maximum, cell)
                else:
                    s.add(cell)

        # Arbitrarily setting lookup values to start from max + 1
        curr = maximum + 1
        self.lookup = dict()
        self.rev_lookup = dict()
        for item in s:
            self.lookup[item] = curr
            self.rev_lookup[curr] = item
            curr += 1

    def show_data(self):
        repr = self.world.get_repr()
        shown = self.get_transformed_data()
        self.mat.set_data(shown)
        for j, row in enumerate(repr):
            for i, z in enumerate(row):
                if type(z) is str:
                    self.ax.text(i, j, z, ha="center", va="center")

    def get_transformed_data(self):
        data = self.world.get_repr()
        return list(
            map(
                lambda row: list(
                    map(
                        lambda cell: self.lookup[cell] if cell in self.lookup else cell,
                        row,
                    )
                ),
                data,
            )
        )

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
