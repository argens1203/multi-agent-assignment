import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import List, Tuple

from core import Grid, GridUtil, Agent
from shared import State


class IVisual(ABC):
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


class Model(IVisual):
    def __init__(self):
        # Metrics
        self.total_reward = 0

        # Agents
        self.agents = []

        # Grid
        self.grid = None

    def bind(self, grid: "Grid"):
        self.grid = grid
        return self

    def add_agent(self, agent: "Agent"):
        self.agents.append(agent)
        return self

    # ---- Public Getter Functions (For Visualisation) ----- #

    def get_agent_info(self) -> List[Tuple[Tuple[int, int], bool]]:
        """
        Output: List of
                - Tuple of:
                    - coordinate: (int, int)
                    - has_item: bool
        """
        has_items = map(lambda agent: agent.has_item(), self.agents)
        return list(zip(self.grid.get_state().get_agent_positions(), has_items))

    def get_untaken_items(self):
        return self.grid.get_state().get_untaken_item_pos()

    def get_max_reward(self):
        return self.max_reward

    def get_size(self):
        return self.grid.get_size()

    def get_target_location(self):
        return self.grid.get_state().get_goal_positions()

    def has_ended(self):
        return self.grid.get_state().is_terminal()

    def get_total_reward(self):
        return sum(map(lambda a: a.get_total_reward(), self.agents))

    # ---- Public Control Functions ----- #
    def reset(self):
        self.total_reward = 0
        self.grid.reset()
        for agent in self.agents:
            agent.reset()
        self.max_reward = GridUtil.calculate_max_reward(self.grid)
