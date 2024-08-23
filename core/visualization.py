import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

from typing import Tuple, TypeAlias

Coordinates: TypeAlias = Tuple[float, float, float, float]


class Visualization:
    def __init__(self, environment, agent):
        self.environment = environment
        self.environment.reset()
        self.agent = agent
        self.fig, self.ax = plt.subplots()

        self.add_ui_elements()
        self.update()

    def add_ui_elements(self):
        # Add button for next step
        self.next_step_button = self.add_button(
            [0.8, 0.01, 0.1, 0.075], "Next Step", self.next_step
        )

        # Add button for reset
        self.reset_button = self.add_button(
            [0.65, 0.01, 0.1, 0.075], "Reset", self.reset
        )

        # Add text box for cumulative reward
        self.reward = self.add_text(
            [0.01, 0.01, 0.2, 0.075], f"Reward: {self.agent.total_reward}"
        )

        # Add text box for max reward
        self.max_reward = self.add_text(
            [0.25, 0.01, 0.2, 0.075],
            f"Max Reward: {self.environment.calculate_max_reward()}",
        )

        self.update()

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

    def draw_grid(self):
        self.ax.clear()
        size = self.environment.size
        for x in range(size):
            for y in range(size):
                rect = patches.Rectangle(
                    (x, y), 1, 1, linewidth=1, edgecolor="black", facecolor="white"
                )
                self.ax.add_patch(rect)
        self.ax.set_xlim(0, size)
        self.ax.set_ylim(size, 0)
        self.ax.set_aspect("equal")

        # Move x-axis labels to the top
        self.ax.xaxis.set_label_position("top")
        self.ax.xaxis.tick_top()

        # Draw target
        tx, ty = self.environment.B_position
        target_patch = patches.Rectangle(
            (tx, ty), 1, 1, linewidth=1, edgecolor="black", facecolor="green"
        )
        self.ax.add_patch(target_patch)

    def reset(self, event):
        self.environment.reset()
        self.agent.total_reward = 0
        self.max_reward.set_text(
            f"Max Reward: {self.environment.calculate_max_reward()}"
        )
        self.update()

    def update(self):
        self.draw_grid()

        # Draw agent
        ax, ay = self.agent.get_position()
        agent_color = "blue" if not self.agent.has_item() else "orange"
        agent_patch = patches.Circle((ax + 0.5, ay + 0.5), 0.3, color=agent_color)
        self.ax.add_patch(agent_patch)

        # Draw item
        if not self.agent.has_item():
            ix, iy = self.environment.item_position
            item_patch = patches.Circle((ix + 0.5, iy + 0.5), 0.2, color="red")
            self.ax.add_patch(item_patch)

        # Update reward text
        self.reward.set_text(f"Reward: {self.agent.total_reward}")

        # Check if the environment is terminal
        if self.environment.is_terminal():
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
            # self.button.disconnect(self.button_connection_id)
        self.fig.canvas.draw()

    def next_step(self, i):
        self.agent.move()
        self.update()

    def animate(self, i):
        if not self.environment.is_terminal():
            self.next_step(1)

    def show(self):
        plt.show()
