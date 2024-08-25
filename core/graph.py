import matplotlib.animation as animation
import matplotlib.pyplot as plt

from .visualization import Controller
from multiprocessing import Process


class Graph:
    def __init__(self, game, fig, axs):
        self.game = game
        self.fig = fig
        if type(axs) == tuple:
            self.ax1, self.ax2 = axs
        else:
            self.ax1, self.ax2 = axs, None

        # self.ani.resume()
        self.controller = Controller(self.game)
        # self.controller.train_once(10000)

        p = Process(target=self.controller.train_once, args=[50000])
        p.start()

        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=100, save_count=100
        )

        self.iteration = 0
        self.iterations = []
        self.losses = []
        self.total_rewards = []

        plt.show()

    def frames(self):
        # print("frame")
        while True:
            yield self.controller.get_stats()

    def draw(self, args):
        # print("draw")
        # print(args)
        # iterations, losses, rewards = args
        # self.iaterations = self.controller.iterations
        # self.losses = self.controller.losses
        # self.total_rewards = self.controller.total_reward

        # self.ax.clear()
        self.plot_training(
            self.controller.iterations,
            self.controller.losses,
            self.controller.total_reward,
        )

    def plot_training(self, iterations, losses, rewards):
        # print(iterations, losses, rewards)
        self.plot_losses(self.ax1, iterations, losses)
        if self.ax2 is not None:
            self.plot_rewards(self.ax2, iterations, rewards)

    def plot_losses(self, ax, iterations, losses):
        # Plotting the loss in the first subplot
        ax.plot(iterations, losses, color="blue", label="Loss")
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

    def plot_rewards(self, ax, iterations, rewards):
        # Plotting the total rewards in the second subplot
        ax.plot(
            iterations,
            rewards,
            label="Reward",
            color="orange",
            marker=",",
        )
        ax.set_title("Reward per game")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
