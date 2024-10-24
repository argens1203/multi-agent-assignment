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


# TODO: print step in text


class IVisual(ABC):

    # Getting Info
    @abstractmethod
    def get_agent_info(self) -> List[Tuple[Tuple[int, int], bool]]:
        pass

    @abstractmethod
    def get_total_reward(self) -> int:
        pass

    @abstractmethod
    def get_min_step(self) -> int:
        pass

    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_goal_positions(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def get_agent_positions(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def has_ended(self) -> bool:
        pass

    # Functions
    @abstractmethod
    def toggle_auto_reset(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    # Trainings/Testing
    @abstractmethod
    def train(self, itr=1):
        pass

    @abstractmethod
    def test(self, itr=1):
        pass

    @abstractmethod
    def train_in_background(self):
        pass

    @abstractmethod
    def test_in_background(self, ep=1000):
        pass


class Visualization:
    def __init__(self, storage: "Storage", grid: "Grid"):
        self.storage = storage
        self.grid = grid
        self.fig, self.ax = plt.subplots()
        self.animating = False

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
        agent_info = self.grid.get_agent_info()
        step_count = self.grid.get_step_count()
        min_step = self.grid.get_min_step()
        return agent_info, step_count, min_step

    def frames(self):
        while True:
            if self.animating:
                self.grid.next()
                yield self.get_info()
            else:
                yield self.get_info()

    # ----- ----- ----- ----- Drawing Functions  ----- ----- ----- ----- #

    def draw(self, args):
        agent_info, step_count, min_step = args

        self.ax.clear()
        self.draw_grid()
        self.draw_agent(agent_info)

        self.step_count.set_text(f"Step: {step_count}")
        self.min_step.set_text(f"Min Step: {min_step}")

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
        tx, ty = self.grid.get_goal_positions()
        target_patch = patches.Rectangle(
            (tx, ty), 1, 1, linewidth=1, edgecolor="black", facecolor="green"
        )
        self.ax.add_patch(target_patch)

    def draw_agent(self, info):
        # Draw agent
        for idx, (pos, type, has_item, step_count) in enumerate(info):
            dx = [0, 0.5, 0, 0.5][idx]
            dy = [0, 0, 0.5, 0.5][idx]
            ax, ay = pos

            if type == 1:
                agent_color = "red" if has_item else "pink"
            if type == 2:
                agent_color = "blue" if has_item else "cyan"

            agent_patch = patches.Rectangle(
                (ax + dx, ay + dy),
                0.5,
                0.5,
                linewidth=1,
                edgecolor="black",
                facecolor=agent_color,
            )
            self.ax.add_patch(agent_patch)
            self.ax.text(
                ax + dx + 0.17,
                ay + dy + 0.33,
                step_count,
                c="yellow",
                ma="center",
                size="large",
                weight="bold",
                # backgroundcolor="white",
            )

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
        self.step_count = self.add_text([0.01, 0.01, 0.2, 0.075], f"Step: 0")

        # Add text box for max reward
        self.min_step = self.add_text(
            [0.25, 0.01, 0.2, 0.075],
            f"Min Step: {self.grid.get_min_step()}",
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

    # ----- ----- ----- ----- Event Handlers  ----- ----- ----- ----- #

    def on_close(self, e):
        self.ani.event_source.stop()

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


class Graph:
    def __init__(self, storage: "Storage", keys=[]):
        self.storage = storage
        self.keys = keys

        self.fig, self.axs = plt.subplots(1, len(self.keys))
        if len(self.keys) == 1:
            self.axs = [self.axs]
        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=100, save_count=100
        )

        plt.show()

    def frames(self):
        while True:
            yield None

    # Compulsory unused argument
    def draw(self, args):
        for i, k in enumerate(self.keys):
            self.plot(
                self.axs[i],
                range(len(getattr(self.storage, k))),  # self.storage.iterations
                getattr(self.storage, k),
                k,
            )

    def plot(self, ax, itr, values, key):
        ax.plot(itr, values, color="blue", label="Loss")
        ax.set_title(key.title())
        ax.set_xlabel("Iteration")
        ax.set_ylabel(key.title())


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
