from typing import TYPE_CHECKING, List, Tuple, Dict, Set

import random

from shared import State, Action
from .cell import *

if TYPE_CHECKING:
    from .agent import Agent


class GridFactory:
    # Getting a random location in a grid, excluding certain locations
    def get_random_pos(
        width: int, height: int, exclude: List[Tuple[int, int]] = []
    ) -> Tuple[int, int]:
        while True:
            position = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            if position not in exclude:
                return position


class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.max_reward = 0

        self.state: Dict[Tuple[int, int], Cell] = (
            {}
        )  # TODO: multiple entities in one cell
        self.lookup: set[Cell] = set()  # Interactive tiles
        self.agents: List["Agent"] = []
        self.agent_positions: List[Tuple[int, int]] = []

        self.init_environment()

    # ----- Init Functions ----- #
    def init_environment(self):
        for x in range(-1, self.width + 1):
            for y in range(-1, self.height + 1):
                if x < 0 or x >= self.width:
                    self.state[(x, y)] = Wall((x, y), (self.width, self.height))
                elif y < 0 or y >= self.height:
                    self.state[(x, y)] = Wall((x, y), (self.width, self.height))
                else:
                    self.state[(x, y)] = Cell((x, y))

    # ----- Core Functions ----- #
    def move(
        self, actions: List["Action"]
    ):  # List of actions, in the same order as self.agents
        # Update agent to temporary location according to move
        temp_positions = [
            self.process_action(action, agent_pos)
            for action, agent_pos in zip(actions, self.agent_positions)
        ]

        # Retreive reward and new location according to Entity.interaction
        reward_new_positions = [
            self.state[(x, y)].interact(agent)
            for agent, (x, y) in zip(self.agents, temp_positions)
        ]
        rewards, new_positions, is_terminal = zip(*reward_new_positions)

        # Update new positions
        self.agent_positions = new_positions

        # Return move results, in the same order as self.agents
        return [
            (reward, self.get_state(), self.get_state().is_terminal())
            for reward in rewards
        ]

    # ----- Private Functions ----- #
    def process_action(
        self, action: List["Action"], agent_position: List[Tuple[int, int]]
    ):
        # Move according to action
        x, y = agent_position
        dx, dy = self.interpret_action(action)
        return x + dx, y + dy

    def interpret_action(self, action: "Action"):
        if action == Action.NORTH:
            return 0, -1
        if action == Action.SOUTH:
            return 0, 1
        if action == Action.EAST:
            return 1, 0
        if action == Action.WEST:
            return -1, 0

    def set_interactive_tiles(self):
        self.lookup.clear()
        used_pos = []

        # TODO: extract repeated code

        # Assign goal to set position
        goal_pos = GridFactory.get_random_pos(self.width, self.height, used_pos)
        # goal_pos = (self.width - 1, self.height - 1)
        goal = Goal(goal_pos)
        self.state[goal_pos] = goal
        self.lookup.add(goal)
        used_pos.append(goal_pos)

        # Assign items to a random position in the remaining tiles
        item_pos = GridFactory.get_random_pos(self.width, self.height, used_pos)
        item = Item(item_pos)
        self.state[item_pos] = item
        self.lookup.add(item)
        used_pos.append(item_pos)

        # Assign agents to random positions
        self.agent_positions = []
        for _ in self.agents:
            agent_pos = GridFactory.get_random_pos(self.width, self.height, used_pos)
            used_pos.append(agent_pos)
            self.agent_positions.append(agent_pos)

        # Future proofing: update agents in case they spwaned on an item
        for agent in self.agents:
            agent.update(State(self.agent_positions, self.lookup))

        self.max_reward = GridUtil.calculate_max_reward(self)

    # ----- Public Functions ----- #
    def reset(self):
        self.init_environment()
        self.set_interactive_tiles()

    def add_agent(self, agent: "Agent"):
        self.agents.append(agent)

    def get_state(self):
        return State(self.agent_positions, self.lookup)

    def get_size(self):
        return self.width, self.height

    def get_max_reward(self):
        return self.max_reward


class GridUtil:
    def calculate_max_reward(grid: Grid):
        # TODO: can only work with one agent and one item ATM
        x1, y1 = grid.get_state().get_agent_positions()[0]
        x2, y2 = grid.get_state().get_item_positions()[0]
        x3, y3 = grid.get_state().get_goal_positions()

        # Manhanttan distance from agent to obj and obj to goal
        dist_to_obj = abs(x1 - x2) + abs(y1 - y2)
        dist_to_goal = abs(x2 - x3) + abs(y2 - y3)

        # +100 for reward and +2 for 2 unneeded mark deduction when stepping on item and goal respectively
        return (dist_to_obj + dist_to_goal) * -1 + 102
