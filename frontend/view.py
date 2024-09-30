import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from matplotlib.widgets import Button
from typing import Tuple, TypeAlias, TYPE_CHECKING

from .v_graph import MLGraph

if TYPE_CHECKING:
    from .model import Model
    from .controller import Controller
    from .c_storage import Storage

Coordinates: TypeAlias = Tuple[float, float, float, float]


class Visualization:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.animating = False

    def bind(self, model: "Model", controller: "Controller", storage: "Storage"):
        self.model = model
        self.controller = controller
        self.storage = storage
        return self

    def show(self):
        assert self.model is not None
        assert self.controller is not None

        self.add_ui_elements()
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.ani = animation.FuncAnimation(
            self.fig, self.draw, frames=self.frames, interval=200, save_count=100
        )
        self.animating = True
        plt.show()

    def get_info(self):
        info = self.model.get_agent_info()
        items = self.model.get_untaken_items()
        tot_reward = self.model.get_total_reward()
        max_reward = self.model.get_max_reward()
        return info, items, tot_reward, max_reward

    def frames(self):
        while True:
            if self.animating:
                self.controller.next()
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
        if self.model.has_ended():
            self.draw_complete()

        # Early return if animating, since animation automatically refreshes canvas
        if self.animating:
            return

        self.fig.canvas.draw()

    def draw_grid(self):
        width, height = self.model.get_size()
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
        tx, ty = self.model.get_target_location()
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
            [0.01, 0.01, 0.2, 0.075], f"Reward: {self.model.get_total_reward()}"
        )

        # Add text box for max reward
        self.max_reward = self.add_text(
            [0.25, 0.01, 0.2, 0.075],
            f"Max Reward: {self.model.get_max_reward()}",
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
            self.controller.test_in_background(episodes)
            self.after_auto_train()

        def blocking_test(e):
            losses = self.controller.test(episodes)
            fig2, ax2 = plt.subplots()
            MLGraph(losses, fig2, ax2).show()

        return blocking_test if blocking else non_blocking_test

    def on_train(self, episodes, blocking=False):
        def blocking_train(e):
            ml_losses = self.controller.train(episodes)
            fig2, ax2 = plt.subplots()
            MLGraph(ml_losses, fig2, ax2).show()

        def non_blocking_train(e):
            self.before_auto_train()
            self.controller.train_in_background()
            self.after_auto_train()

        return blocking_train if blocking else non_blocking_train

    def on_auto_reset(self, event):
        auto_reset_is_on = self.controller.toggle_auto_reset()
        if auto_reset_is_on:
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
        self.model.reset()
        self.draw(self.get_info())

    def on_next(self, e):
        self.controller.next()
        self.draw(self.get_info())

    # ----- ----- ----- ----- Helper Functions  ----- ----- ----- ----- #

    def before_auto_train(self):
        self.animating = False
        self.controller.reset()

        self.toggle_anim_btn.label.set_text("Anim\nOff")
        self.draw(self.get_info())

    def after_auto_train(self):
        self.animating = True
        self.controller.reset()

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
