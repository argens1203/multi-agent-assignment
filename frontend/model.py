from abc import ABC, abstractmethod
from typing import List, Tuple

from core import Grid, Agent


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

    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_target_location(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def has_ended(self) -> bool:
        pass


class Model(IVisual):
    def __init__(self):
        # Agents
        self.agents = []

        # Grid
        self.grid: "Grid" = None

    def set_grid(self, grid: "Grid"):
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
        return list(zip(self.grid.get_agent_positions(), has_items))

    def get_untaken_items(self):
        return self.grid.get_untaken_item_pos()

    def get_total_reward(self):
        return sum(map(lambda a: a.get_total_reward(), self.agents))

    def get_max_reward(self):
        return self.grid.get_max_reward()

    def get_size(self):
        return self.grid.get_size()

    def get_target_location(self):
        return self.grid.get_goal_positions()

    def has_ended(self):
        return self.grid.goal.has_reached()

    # ---- Public Control Functions ----- #
    def reset(self):
        self.grid.reset()
        for agent in self.agents:
            agent.reset()
