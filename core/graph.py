import matplotlib.animation as animation
import matplotlib.pyplot as plt

from .visualization import Controller


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
        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=100, save_count=100
        )

        self.iteration = 0
        self.iterations = []
        self.losses = []
        self.total_rewards = []

        plt.show()

    def frames(self):
        while True:
            yield self.controller.train_once()

    def draw(self, args):
        itr, loss, reward_earned = args
        self.iterations.append(itr)
        self.losses.append(loss)
        self.total_rewards.append(reward_earned)

        # self.ax.clear()
        self.plot_training()

    def plot_training(self):
        self.plot_losses(self.ax1)
        if self.ax2 is not None:
            self.plot_rewards(self.ax2)

    def plot_losses(self, ax):
        # Plotting the loss in the first subplot
        ax.plot(self.iterations, self.losses, color="blue", marker="o", label="Loss")
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

    def plot_rewards(self, ax):
        # Plotting the total rewards in the second subplot
        ax.plot(
            self.iterations,
            self.total_rewards,
            label="Reward",
            color="orange",
            marker="o",
        )
        ax.set_title("Reward per game")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
