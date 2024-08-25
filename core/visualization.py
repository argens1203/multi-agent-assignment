import matplotlib.pyplot as plt
import matplotlib.patches as patches

# import plotly
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation
from multiprocessing import Array
from typing import Tuple, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from .game import Game

Coordinates: TypeAlias = Tuple[float, float, float, float]

# plotly.offline.init_notebook_mode(connected=True)


class Controller(object):
    def __init__(self, game):
        self.game = game
        self.timeout = 0.5
        self.auto_reset = True
        self.itr = 0

        self.iterations = Array("i", range(50000))
        self.losses = Array("i", 50000)

    def get_info(self):
        info = self.game.get_agent_info()
        items = self.game.get_untaken_items()
        tot_reward = self.game.get_total_reward()
        max_reward = self.game.get_max_reward()
        return info, items, tot_reward, max_reward

    def set_timeout(self, timeout):
        self.timeout = timeout

    def toggle_auto_reset(self):
        self.auto_reset = not self.auto_reset
        return self.auto_reset

    def next(self):
        if self.game.has_ended() and self.auto_reset:
            self.game.reset()
            # self.game.step(learn=False)
        self.game.step(learn=False)
        return self.get_info()

    def __call__(self, learning=False):
        if learning:
            return self.train_once()
        else:
            return self.next()

    def train_once(self, itr=1):
        for i in range(itr):
            (
                loss,
                reward,
                max_reward,
            ) = self.game.train_agent_once()
            self.losses[self.itr] = loss
            self.itr += 1


class Visualization:
    def __init__(self, game: "Game", fig, ax):
        self.game = game
        self.is_stopping = False
        self.timer = None
        self.game.reset()
        self.speed = 1
        self.fig = fig
        self.ax = ax

        self.add_ui_elements()
        # self.update()
        self.controller = Controller(self.game)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=200, save_count=100
        )

        self.animating = True
        # self.ani.pause()

        plt.show()

    def frames(self):
        while True:
            yield self.controller(learning=False)

    def draw(self, args):
        info, items, tot_reward, max_reward = args

        self.ax.clear()
        self.draw_grid()
        self.draw_agent(info)
        self.draw_item(items)

        self.reward.set_text(f"Reward: {tot_reward}")
        self.max_reward.set_text(f"Reward: {max_reward}")

        # Check if the environment is terminal
        if self.game.has_ended():
            self.draw_complete()
        if not self.animating:
            self.fig.canvas.draw()

    def draw_grid(self):
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
        self.init_text()

    def init_buttons(self):
        # Add button for next step
        self.next_step_btn = self.add_button(
            [0.85, 0.01, 0.12, 0.075], "Next Step", self.on_next
        )
        # Add button for reset
        self.reset_btn = self.add_button(
            [0.85, 0.11, 0.12, 0.075], "Reset", self.on_reset
        )
        # Add button for animation on/off
        self.toggle_anim_btn = self.add_button(
            [0.85, 0.21, 0.12, 0.075], "Anim\nOn", self.on_toggle_anim
        )
        # Add button for auto reset on/off
        self.toggle_auto_reset_btn = self.add_button(
            [0.85, 0.31, 0.12, 0.075], "Auto Reset\nOn", self.on_auto_reset
        )

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
        # axis = self.ax
        button = Button(axis, text)
        button.on_clicked(on_click)

        return button

    def add_text(self, coordinates: Coordinates, text):
        axis = plt.axes(coordinates)
        # axis = self.ax
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
    def one_step(self):
        self.draw(self.controller.next())

    def stop_anim(self):
        pass

    def start_anim(self):
        pass

    # ----- ----- ----- ----- Event Handlers  ----- ----- ----- ----- #

    def on_toggle_anim(self, event):
        if self.animating:
            self.ani.pause()
            self.toggle_anim_btn.label.set_text("Anim\nOff")
        else:
            self.ani.resume()
            self.toggle_anim_btn.label.set_text("Anim\nOn")

        self.animating = not self.animating
        plt.show()

    def on_auto_reset(self, event):
        auto_reset_is_on = self.controller.toggle_auto_reset()
        if auto_reset_is_on:
            self.toggle_auto_reset_btn.label.set_text("Auto Reset\nOn")
        else:
            self.toggle_auto_reset_btn.label.set_text("Auto Reset\nOff")
        plt.show()

    def on_reset(self, event):
        self.game.reset()
        self.draw(self.controller.get_info())

    def on_next(self, e):
        self.draw(self.controller.next())

    def on_close(self, e):
        pass

    # ----- ----- ----- ----- Plot Metrics  ----- ----- ----- ----- #
    def plot_training(results):
        # print(results)
        iterations = [t[0] for t in results]
        losses = [t[1] for t in results]
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
