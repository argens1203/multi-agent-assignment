import matplotlib.pyplot as plt

from .grid import GridWorld as Environment, GridUtil
from .agent import Agent
from .visualization import Visualization
from .state import State

debug = False

class Game:
    def __init__(self):
        self.total_reward = 0
        self.agent = [Agent(i, State.get_possible_states(), State.get_possible_actions()) for i in range(1)]
        self.env = Environment(8, 8)
        self.env.add_agents(self.agent)
        self.env.reset()

    def run(self):
        training_record = self.train_agent(50000)
        plot_training(training_record)
        vis = Visualization(self)
        vis.reset(None)
        vis.show()

    def train_agent(self, episodes):
        training_record = []
        for _ in range(episodes):
            self.env.reset()
            self.total_reward = 0
            max_reward = GridUtil.calculate_max_reward(self.env)

            while not self.env.is_terminal():
                self.step()

            loss = max_reward - self.total_reward
            if debug:
                print(
                    f"Episode {_} completed with total reward: {self.total_reward},max_reward:{max_reward}, loss:{loss}"
                )
            training_record.append([_, max_reward, self.total_reward, loss])

        # print("Training complete")
        return training_record

    def get_agents(self):
        return self.agent

    def get_agent_positions(self):
        return self.env.get_agent_positions()

    def get_untaken_items(self):
        return self.env.get_untaken_item_pos()

    def get_max_reward(self):
        return GridUtil.calculate_max_reward(self.env)

    def get_size(self):
        return self.env.get_size()

    def get_target_location(self):
        return self.env.get_goal_positions()

    def has_ended(self):
        return self.env.is_terminal()

    def reset(self):
        self.total_reward = 0
        self.env.reset()

    def step(self, learn=True):
        if self.env.is_terminal():
            return
        state = self.env.get_state()
        
        actions = [agent.choose_action(state) for agent in self.agent] # TODO: disable epsilon whent testing
        results = self.env.move(actions)

        for action, (reward, next_state, terminal), agent in zip(actions, results, self.agent):
            self.total_reward += reward
            if learn:
                agent.update_learn(state, action, reward, next_state, terminal)
            else:
                agent.update(next_state)


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
