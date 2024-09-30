from abc import ABC, abstractmethod
from typing import List, Tuple

from core import Grid, Agent


class Model:
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
