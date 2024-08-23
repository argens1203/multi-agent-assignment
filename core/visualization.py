import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

from typing import Tuple, TypeAlias

Coordinates: TypeAlias = Tuple[float, float, float, float]


class Visualization:
    def __init__(self, game, environment, agent):
        self.game = game
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
        self.agent.set_state(self.environment.get_state())
        self.max_reward.set_text(
            f"Max Reward: {self.environment.calculate_max_reward()}"
        )
        self.update()

    def update(self):
        self.draw_grid()

        self.draw_agent(self.game.get_agents())
        self.draw_item(self.game.get_untaken_items())
        self.reward.set_text(f"Reward: {self.agent.total_reward}")

        # Check if the environment is terminal
        if self.environment.is_terminal():
            self.draw_complete()
        self.fig.canvas.draw()

    def draw_agent(self, agents):
        # Draw agent
        for agent in agents:
            print(agent.position)
            ax, ay = agent.position
            agent_color = "blue" if not agent.has_item else "orange"
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

    def next_step(self, i):
        self.agent.move()
        self.update()

    def animate(self, i):
        if not self.environment.is_terminal():
            self.next_step(1)

    def show(self):
        plt.show()
