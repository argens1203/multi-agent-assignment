import matplotlib.pyplot as plt

from typing import List, Tuple

from .grid import Grid, GridUtil
from .agent import Agent
from .state import State


class Game:
    def __init__(self):
        # Parameters
        self.width = 5
        self.height = 5

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
        self.reset()

    def train_one_game(self):
        self.reset()
        self.total_reward = 0
        max_reward = GridUtil.calculate_max_reward(self.grid)

        while not self.grid.get_state().is_terminal():
            self.step()

        loss = max_reward - self.total_reward
        return loss, self.total_reward, max_reward

    # ---- Public Getter Functions (For Visualisation) ----- #

    def get_agent_info(self) -> List[Tuple[Tuple[int, int], bool]]:
        """
        Output: List of
                - Tuple of:
                    - coordinate: (int, int)
                    - has_item: bool
        """
        has_items = map(lambda agent: agent.has_item(), self.agent)
        return list(zip(self.grid.get_state().get_agent_positions(), has_items))

    def get_untaken_items(self):
        return self.grid.get_state().get_untaken_item_pos()

    def get_max_reward(self):
        return self.max_reward

    def get_size(self):
        return self.width, self.height

    def get_target_location(self):
        return self.grid.get_state().get_goal_positions()

    def has_ended(self):
        return self.grid.get_state().is_terminal()

    def get_total_reward(self):
        return self.total_reward

    # ---- Public Control Functions ----- #
    def reset(self):
        self.total_reward = 0
        self.grid.reset()
        for agent in self.agent:
            agent.reset()
        self.max_reward = GridUtil.calculate_max_reward(self.grid)

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
