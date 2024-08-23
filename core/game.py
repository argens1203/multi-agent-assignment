import matplotlib.pyplot as plt
import itertools

from .grid import GridWorld as Environment, Action, GridUtil
from .agent import Agent
from .visualization import Visualization

debug = False


class Game:
    def __init__(self):
        self.total_reward = 0

    def run(self):
        self.env = Environment(size=8)
        self.agent = self.get_agent()
        training_record = self.train_agent(50000)
        plot_training(training_record)
        vis = Visualization(self)
        vis.reset(None)
        vis.show()

    def get_agent(self):
        # Generate all possible states
        positions = [(x, y) for x in range(8) for y in range(8)]
        has_items = [True, False]
        possible_states = itertools.product(positions, positions, has_items)

        # Generate possible actions
        possible_actions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
        
        return Agent(possible_states, possible_actions)

    def train_agent(self, episodes):
        training_record = []
        for _ in range(episodes):
            self.env.reset()
            state = self.env.get_state()
            max_reward = GridUtil.calculate_max_reward(self.env)

            total_reward = 0
            while not self.env.is_terminal():
                action = self.agent.choose_action(state)
                next_state, reward, terminal = self.env.move(action)
                self.agent.perceive(state, action, reward, next_state, terminal)

                state = next_state
                total_reward += reward

            loss = max_reward - total_reward
            if debug:
                print(
                    f"Episode {_} completed with total reward: {total_reward},max_reward:{max_reward}, loss:{loss}"
                )
            training_record.append([_, max_reward, total_reward, loss])

        print("Training complete")
        return training_record

    def get_agents(self):
        return [self.agent]

    def get_untaken_items(self):
        pos, has_item = self.agent.get_props()
        return [] if has_item else self.env.get_item_positions()

    def get_max_reward(self):
        return GridUtil.calculate_max_reward(self.env)

    def get_size(self):
        return self.env.size

    def get_target_location(self):
        return self.env.get_goal_positions()

    def has_ended(self):
        return self.env.is_terminal()

    def reset(self):
        self.env.reset()
        self.total_reward = 0
        self.agent.set_state(self.env.get_state())

    def step(self):
        if self.env.is_terminal():
            return
        state = self.env.get_state()
        action = self.agent.choose_action(
            state
            # TODO: disable epsilon whent testing
        )  # TODO: Should only choose among valid moves?
        next_state, reward, terminal = self.env.move(action)
        self.agent.set_state(next_state)
        self.total_reward += reward


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
