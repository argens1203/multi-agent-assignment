import matplotlib.animation as animation
import matplotlib.pyplot as plt

from .controller import Controller
from multiprocessing import Process


class Graph:
    def __init__(self, controller, fig, ax):
        self.controller = controller
        self.fig = fig
        self.ax = ax

        self.controller = controller
        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=100, save_count=100
        )

        plt.show()

    def frames(self):
        while True:
            yield None

    def draw(self, args):
        self.plot_losses(
            self.ax,
            self.controller.iterations,
            self.controller.losses,
        )

    def plot_losses(self, ax, iterations, losses):
        # Plotting the loss in the first subplot
        ax.plot(iterations, losses, color="blue", label="Loss")
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
