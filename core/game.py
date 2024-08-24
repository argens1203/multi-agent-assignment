import matplotlib.pyplot as plt

from .grid import Grid, GridUtil
from .agent import Agent
from .visualization import Visualization
from .state import State

debug = False


class Game:
    def __init__(self):
        # Parameters
        self.width = 8
        self.height = 8

        # Metrics
        self.total_reward = 0

        # Agents
        self.agent = [
            Agent(
                idx,
                State.get_possible_states(self.width, self.height),
                State.get_possible_actions(),
            )
            for idx in range(1)
        ]

        # Grid
        self.grid = Grid(self.width, self.height)
        self.grid.add_agents(self.agent)
        self.grid.reset()

    # This is the main function to be called for external
    def run(self):
        training_record = self.train_agent(50000)
        Visualization.plot_training(training_record)
        vis = Visualization(self)
        vis.reset(None)
        vis.show()

    def train_agent(self, episodes):
        training_record = []
        for _ in range(episodes):
            self.grid.reset()
            self.total_reward = 0
            max_reward = GridUtil.calculate_max_reward(self.grid)

            while not self.grid.get_state().is_terminal():
                self.step()

            loss = max_reward - self.total_reward
            if debug:
                print(
                    f"Episode {_} completed with total reward: {self.total_reward},max_reward:{max_reward}, loss:{loss}"
                )
            training_record.append([_, max_reward, self.total_reward, loss])
        return training_record

    # ---- Public Getter Functions (For Visualisation) ----- #
    def get_agents(self):
        return self.agent

    def get_agent_positions(self):
        return self.grid.get_state().get_agent_positions()

    def get_untaken_items(self):
        return self.grid.get_state().get_untaken_item_pos()

    def get_max_reward(self):
        return GridUtil.calculate_max_reward(self.grid)

    def get_size(self):
        return self.width, self.height

    def get_target_location(self):
        return self.grid.get_state().get_goal_positions()

    def has_ended(self):
        return self.grid.get_state().is_terminal()

    # ---- Public Control Functions ----- #
    def reset(self):
        self.total_reward = 0
        self.grid.reset()

    def step(self, learn=True):
        if self.grid.get_state().is_terminal():
            return
        state = self.grid.get_state()

        actions = [agent.choose_action(state, explore=False) for agent in self.agent]
        results = self.grid.move(actions)

        for action, (reward, next_state, terminal), agent in zip(
            actions, results, self.agent
        ):
            self.total_reward += reward
            if learn:
                agent.update_learn(state, action, reward, next_state, terminal)
            else:
                agent.update(next_state)
