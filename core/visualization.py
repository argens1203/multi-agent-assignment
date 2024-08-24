import matplotlib.pyplot as plt
import matplotlib.patches as patches
import threading
import plotly
from plotly.offline import iplot
import plotly.graph_objs as go
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation

from typing import Tuple, TypeAlias, TYPE_CHECKING
import time
import random

if TYPE_CHECKING:
    from .game import Game

Coordinates: TypeAlias = Tuple[float, float, float, float]

plotly.offline.init_notebook_mode(connected=True)


class RegrMagic(object):
    def __init__(self, game):
        self.game = game
        self.timeout = 0.5

    def get_info(self):
        info = self.game.get_agent_info()
        items = self.game.get_untaken_items()
        tot_reward = self.game.total_reward
        return info, items, tot_reward

    def set_timeout(self, timeout):
        self.timeout = timeout

    def next(self):
        self.game.step(learn=False)
        return self.get_info()

    def __call__(self):
        # time.sleep(self.timeout)
        return self.next()


class Visualization:
    def __init__(self, game: "Game"):
        self.game = game
        self.is_stopping = False
        self.timer = None
        self.game.reset()
        self.speed = 1

        self.fig, self.ax = plt.subplots()
        self.add_ui_elements()
        # self.update()
        self.controller = RegrMagic(self.game)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=100, save_count=100
        )

        self.animating = False
        self.ani.pause()

        plt.show()

    def frames(self):
        while True:
            yield self.controller()

    def draw(self, args):
        info, items, tot_reward = args

        self.ax.clear()
        self.draw_grid()
        self.draw_agent(info)
        self.draw_item(items)
        self.reward.set_text(f"Reward: {tot_reward}")

        # Check if the environment is terminal
        if self.game.has_ended():
            self.draw_complete()
        if not self.animating:
            self.fig.canvas.draw()

    def draw_grid(self):
        # self.ax.clear()
        width, height = self.game.get_size()
        for x in range(width):
            for y in range(height):
                rect = patches.Rectangle(
                    (x, y), 1, 1, linewidth=1, edgecolor="black", facecolor="white"
                )
                self.ax.add_patch(rect)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.set_aspect("equal")

        # Move x-axis labels to the top
        self.ax.xaxis.set_label_position("top")
        self.ax.xaxis.tick_top()

        # Draw target
        tx, ty = self.game.get_target_location()
        target_patch = patches.Rectangle(
            (tx, ty), 1, 1, linewidth=1, edgecolor="black", facecolor="green"
        )
        self.ax.add_patch(target_patch)

    def draw_agent(self, info):
        # Draw agent
        for pos, has_item in info:
            ax, ay = pos
            agent_color = "blue" if not has_item else "orange"
            agent_patch = patches.Circle((ax + 0.5, ay + 0.5), 0.3, color=agent_color)
            self.ax.add_patch(agent_patch)

    def draw_item(self, items):
        for item in items:
            ix, iy = item
            item_patch = patches.Circle((ix + 0.5, iy + 0.5), 0.2, color="red")
            self.ax.add_patch(item_patch)

    def draw_complete(self):
        self.ax.text(
            0.5,
            0.5,
            "Complete",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
            fontsize=20,
            color="red",
        )

    # ----- ----- ----- ----- Render UI Element  ----- ----- ----- ----- #

    def add_ui_elements(self):
        self.init_buttons()
        self.init_slider()
        self.init_text()
        # self.update()

    def init_buttons(self):
        # Add button for next step
        self.next_step_button = self.add_button(
            [0.85, 0.01, 0.1, 0.075], "Next Step", self.next
        )

        # Add button for reset
        self.reset_button = self.add_button(
            [0.85, 0.11, 0.1, 0.075], "Reset", self.reset
        )
        # Add button for reset
        self.animate_button = self.add_button(
            [0.85, 0.21, 0.1, 0.075], "Animate", self.start_animation
        )
        # Add button for reset
        self.stop_button = self.add_button([0.85, 0.31, 0.1, 0.075], "Stop", self.stop)

    def init_slider(self):
        plt.subplots_adjust(bottom=0.15)
        axspeed = plt.axes([0.175, 0.08, 0.6, 0.03])
        self.sspeed = Slider(axspeed, "Speed", 0.5, 5, valinit=2.0)
        self.sspeed.on_changed(self.on_speed_update)

    def init_text(self):
        # Add text box for cumulative reward
        self.reward = self.add_text(
            [0.01, 0.01, 0.2, 0.075], f"Reward: {self.game.total_reward}"
        )

        # Add text box for max reward
        self.max_reward = self.add_text(
            [0.25, 0.01, 0.2, 0.075],
            f"Max Reward: {self.game.get_max_reward()}",
        )

    def add_button(self, coordinates: Coordinates, text, on_click):
        axis = plt.axes(coordinates)
        button = Button(axis, text)
        button.on_clicked(on_click)

        return button

    def add_text(self, coordinates: Coordinates, text):
        axis = plt.axes(coordinates)
        textbox = axis.text(
            0.5,
            0.5,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
            fontsize=12,
        )
        axis.axis("off")
        return textbox

    # ----- ----- ----- ----- Render Main Board  ----- ----- ----- ----- #

    # def update(self):
    #     self.draw_grid()

    #     self.draw_agent(self.game.get_agent_info())
    #     self.draw_item(self.game.get_untaken_items())
    #     self.reward.set_text(f"Reward: {self.game.total_reward}")

    #     # Check if the environment is terminal
    #     if self.game.has_ended():
    #         self.draw_complete()

    #     self.fig.canvas.draw()

    # def foo(self):
    #     self.one_step()
    #     if not (self.is_stopping):
    #         self.timer = threading.Timer(self.speed, self.foo)
    #         self.timer.start()

    def one_step(self):
        self.draw(self.controller.next())

    def stop_anim(self):
        pass

    def start_anim(self):
        pass

    # ----- ----- ----- ----- Event Handlers  ----- ----- ----- ----- #
    def start_animation(self, event):
        self.animating = True
        self.ani.resume()

    def stop(self, event):
        print("stop")
        self.animating = False
        self.ani.pause()

    def reset(self, event):
        print("reset")
        # self.stop_anim()
        self.game.reset()
        self.draw(self.controller.get_info())
        # self.max_reward.set_text(f"Max Reward: {self.game.get_max_reward()}")
        # self.update()

    def next(self, e):
        print("next")
        self.draw(self.controller.next())
        # self.stop_anim()
        # self.one_step()

    def on_close(self, e):
        print("on_close")
        pass
        # self.stop_anim()

    def on_speed_update(self, val: float | None):
        if val:
            self.controller.set_timeout(1 / val)
        # if val:
        #     self.speed = 1 / val

    def show(self):
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        plt.show()

    # ----- ----- ----- ----- Plot Metrics  ----- ----- ----- ----- #
    def plot_training(results):
        iterations = [t[0] for t in results]
        losses = [t[3] for t in results]
        total_rewards = [t[2] for t in results]

        # Create a figure with 1 row and 2 columns of subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plotting the loss in the first subplot
        ax1.plot(iterations, losses, marker="o", label="Loss")
        ax1.set_title("Iteration vs Loss")
        ax1.set_xlabel("Iteration Number")
        ax1.set_ylabel("Loss")

        # Plotting the total rewards in the second subplot
        ax2.plot(
            iterations, total_rewards, label="Total Reward", color="orange", marker="o"
        )
        ax2.set_title("Iteration vs Total Reward")
        ax2.set_xlabel("Iteration Number")
        ax2.set_ylabel("Total Reward")

        # Display the plots
        plt.tight_layout()
        plt.show()


    def plot_loss(results):
        iterations = [t[0] for t in results]
        losses = [t[1] for t in results]

        # Create a figure with 1 row and 2 columns of subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plotting the loss in the first subplot
        ax1.plot(iterations, losses, marker="o", label="Loss")
        ax1.set_title("Iteration vs Loss")
        ax1.set_xlabel("Iteration Number")
        ax1.set_ylabel("Loss")

        # Display the plots
        plt.tight_layout()
        plt.show()
