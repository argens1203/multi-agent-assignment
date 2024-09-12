import matplotlib.animation as animation
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, controller, fig, axs):
        self.controller = controller
        self.fig = fig
        self.ax1, self.ax2 = axs

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
            self.ax1,
            self.controller.iterations,
            self.controller.losses,
        )
        self.plot_epsilon(
            self.ax2,
            self.controller.iterations,
            self.controller.epsilon,
        )

    def plot_losses(self, ax, iterations, loss):
        # Plotting the loss in the first subplot
        ax.plot(iterations, loss, color="blue", label="Loss")
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

    def plot_epsilon(self, ax, iterations, epsilon):
        # Plotting the loss in the first subplot
        ax.plot(iterations, epsilon, color="blue", label="Loss")
        ax.set_title("Epsilon")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Epsilon")


class TestGraph:
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
            self.controller.test_loss,
        )

    def plot_losses(self, ax, iterations, loss):
        # Plotting the loss in the first subplot
        ax.plot(iterations, loss, color="blue", label="Loss")
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
