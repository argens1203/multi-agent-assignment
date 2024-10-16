import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from matplotlib.widgets import Button
from typing import Tuple, TypeAlias, TYPE_CHECKING, List
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    # from .controller import Controller
    from core import Grid, Storage

Coordinates: TypeAlias = Tuple[float, float, float, float]


class IVisual(ABC):

    # Getting Info

    @abstractmethod
    def get_agent_info(self) -> List[Tuple[Tuple[int, int], bool]]:
        pass

    @abstractmethod
    def get_untaken_items(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def get_total_reward(self) -> int:
        pass

    @abstractmethod
    def get_max_reward(self) -> int:
        pass

    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_goal_positions(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def has_ended(self) -> bool:
        pass

    # Functions

    @abstractmethod
    def toggle_auto_reset(self):
        pass

    @abstractmethod
    def train(self, itr=1):
        pass

    @abstractmethod
    def test(self, itr=1):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def test_in_background(self, ep=1000):
        pass

    @abstractmethod
    def train_in_background(self):
        pass

    @abstractmethod
    def next(self):
        pass


class Visualization:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.animating = False

    def bind(self, storage: "Storage", grid: "Grid"):
        self.storage = storage
        self.grid = grid
        return self

    def show(self):
        assert self.grid is not None

        self.add_ui_elements()
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=200, save_count=100
        )
        self.animating = True
        plt.show()

    def get_info(self):
        agents = self.grid.get_agent_info()
        items = self.grid.get_untaken_items()
        tot_reward = self.grid.get_total_reward()
        max_reward = self.grid.get_max_reward()
        return agents, items, tot_reward, max_reward

    def frames(self):
        while True:
            if self.animating:
                self.grid.next()
                yield self.get_info()
            else:
                yield self.get_info()

    # ----- ----- ----- ----- Drawing Functions  ----- ----- ----- ----- #

    def draw(self, args):
        info, items, tot_reward, max_reward = args

        self.ax.clear()
        self.draw_grid()
        self.draw_agent(info)
        self.draw_item(items)

        self.reward.set_text(f"Reward: {tot_reward}")
        self.max_reward.set_text(f"Max Reward: {max_reward}")

        # Check if the environment is terminal
        if self.grid.has_ended():
            self.draw_complete()

        # Early return if animating, since animation automatically refreshes canvas
        if self.animating:
            return

        self.fig.canvas.draw()

    def draw_grid(self):
        width, height = self.grid.get_size()
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
        # TODO: cater multiple goals
        tx, ty = self.grid.get_goal_positions()
        target_patch = patches.Rectangle(
            (tx, ty), 1, 1, linewidth=1, edgecolor="black", facecolor="green"
        )
        self.ax.add_patch(target_patch)

    def draw_agent(self, info):
        # Draw agent
        for pos, type, has_item in info:
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
        self.toggle_anim_btn = None
        self.toggle_auto_reset_btn = None

        btn_template = [
            ("Next Step", self.on_next),
            ("Reset", self.on_reset),
            ("Anim\nOn", self.on_toggle_anim, "toggle_anim_btn"),
            ("Auto Reset\nOn", self.on_auto_reset, "toggle_auto_reset_btn"),
            ("Train 2500", self.on_train(2500, blocking=True)),
            ("Train Graph", self.on_show_graph),
            ("Test", self.on_test(100, blocking=True)),
        ]
        self.buttons = []
        x, y, w, h = 0.85, 0.01, 0.12, 0.075
        for template in btn_template:
            ref = None
            try:
                label, cb, ref = template
            except:
                label, cb = template
            self.buttons.append(self.add_button([x, y, w, h], label, cb))
            y += 0.1
            if ref:
                setattr(self, ref, self.buttons[-1])

    def init_text(self):
        # Add text box for cumulative reward
        self.reward = self.add_text(
            [0.01, 0.01, 0.2, 0.075], f"Reward: {self.grid.get_max_reward()}"
        )

        # Add text box for max reward
        self.max_reward = self.add_text(
            [0.25, 0.01, 0.2, 0.075],
            f"Max Reward: {self.grid.get_max_reward()}",
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

    # ----- ----- ----- ----- Event Handlers  ----- ----- ----- ----- #

    def on_close(self, e):
        pass

    def on_show_graph(self, e):
        fig2, ax2 = plt.subplots()
        MLGraph(self.storage.ml_losses, fig2, ax2).show()

    def on_test(self, episodes, blocking=False):
        def non_blocking_test(e):
            self.before_auto_train()
            self.grid.test_in_background(episodes)
            self.after_auto_train()

        def blocking_test(e):
            losses = self.grid.test(episodes)
            fig2, ax2 = plt.subplots()
            MLGraph(losses, fig2, ax2).show()

        return blocking_test if blocking else non_blocking_test

    def on_train(self, episodes, blocking=False):
        def blocking_train(e):
            ml_losses = self.grid.train(episodes)
            fig2, ax2 = plt.subplots()
            MLGraph(ml_losses, fig2, ax2).show()

        def non_blocking_train(e):
            self.before_auto_train()
            self.grid.train_in_background()
            self.after_auto_train()

        return blocking_train if blocking else non_blocking_train

    def on_auto_reset(self, event):
        is_on = self.grid.toggle_auto_reset()
        if is_on:
            self.toggle_auto_reset_btn.label.set_text("Auto Reset\nOn")
        else:
            self.toggle_auto_reset_btn.label.set_text("Auto Reset\nOff")
        plt.show()

    def on_toggle_anim(self, event):
        if self.animating:
            self.toggle_anim_btn.label.set_text("Anim\nOff")
        else:
            self.toggle_anim_btn.label.set_text("Anim\nOn")

        self.animating = not self.animating
        plt.show()

    def on_reset(self, event):
        self.grid.reset()
        self.draw(self.get_info())

    def on_next(self, e):
        self.grid.next()
        self.draw(self.get_info())

    # ----- ----- ----- ----- Helper Functions  ----- ----- ----- ----- #

    def before_auto_train(self):
        self.animating = False
        self.grid.reset()

        self.toggle_anim_btn.label.set_text("Anim\nOff")
        self.draw(self.get_info())

    def after_auto_train(self):
        self.animating = True
        self.grid.reset()

        self.toggle_anim_btn.label.set_text("Anim\nOn")
        self.draw(self.get_info())

    # ----- ----- ----- ----- Plot Metrics  ----- ----- ----- ----- #
    def plot_training(results):
        iterations, losses, total_rewards = results
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
        ax2.set_title("Epsilon decay across iteration")
        ax2.set_xlabel("Iteration Number")
        ax2.set_ylabel("Epsilon")

        # Display the plots
        plt.tight_layout()
        plt.show()


import matplotlib.animation as animation
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING


class Graph:
    def __init__(self, storage: "Storage", fig, axs):
        self.storage = storage
        self.fig = fig
        self.ax1, self.ax2 = axs

        self.storage = storage
        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=100, save_count=100
        )

        plt.show()

    def frames(self):
        while True:
            yield None

    # Compulsory unused argument
    def draw(self, args):
        self.plot_losses(
            self.ax1,
            self.storage.iterations,
            self.storage.losses,
        )
        self.plot_epsilon(
            self.ax2,
            self.storage.iterations,
            self.storage.epsilon,
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
    def __init__(self, storage, fig, ax):
        self.storage = storage
        self.fig = fig
        self.ax = ax

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
            self.storage.iterations,
            self.storage.test_loss,
        )

    def plot_losses(self, ax, iterations, loss):
        # Plotting the loss in the first subplot
        ax.plot(iterations, loss, color="blue", label="Loss")
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")


class MLGraph:
    def __init__(self, ml_losses, fig, ax):
        ax.plot(range(len(ml_losses)), ml_losses, label="Loss")
        # ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        pass

    def show(self):

        # Display the plots
        plt.tight_layout()
        plt.show()
        pass
