import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Dict, Set

import random

from constants import dtype, state_size, side

if TYPE_CHECKING:
    from .agent import Agent


class IInteractable(ABC):
    @abstractmethod
    def interact(self, agent: "Agent") -> Tuple[int, Tuple[int, int]]:
        pass


class Cell:
    def __init__(self, pos):
        x, y = pos
        self.x = x
        self.y = y

    # returns: score delta, (new_coordinate_x, new_coordinate_y)
    def interact(self, agent: "Agent") -> Tuple[int, Tuple[int, int]]:
        return -1, (self.x, self.y), False


class Goal(Cell):
    def __init__(self, pos):
        super().__init__(pos)
        self.reached = False

    def interact(self, agent: "Agent"):
        if agent.has_item() and not self.reached:
            self.reached = True
            return 50, (self.x, self.y), True
        else:
            return -1, (self.x, self.y), False

    def has_reached(self):
        return self.reached


class Item(Cell):
    def __init__(self, pos):
        super().__init__(pos)
        self.taken = False

    def interact(self, agent: "Agent"):
        if not self.taken and not agent.has_item():
            agent.set_has_item(True)
            self.taken = True
            return 50, (self.x, self.y), False

        return -1, (self.x, self.y), False

    def get_pos(self):
        return self.x, self.y


class Wall(Cell):
    def __init__(self, pos, dimensions):
        super().__init__(pos)

        width, height = dimensions
        x, y = pos

        self.new_x = min(width - 1, max(0, x))
        self.new_y = min(height - 1, max(0, y))

    def interact(self, agent: "Agent"):
        return -10, (self.new_x, self.new_y), False


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
    def step(self, learn=True):
        if self.goal.has_reached():
            return

        loss = 0

        for idx, agent in enumerate(self.agents):
            state = self.extract_state(idx)
            action = agent.choose_action(state, explore=learn)
            reward, next_state, terminal = self.move(idx, action)

            if learn:
                loss += agent.update_learn(
                    state,
                    action,
                    reward,
                    next_state,
                    terminal,
                )
            else:
                agent.update(reward)
        return loss if learn else None

    def move(
        self, idx: int, action: Tuple[int, int]
    ):  # List of actions, in the same order as self.agents
        # Update agent to temporary location according to move
        temp_positions = self.process_action(action, self.agent_positions[idx])

        # Retreive reward and new location according to Entity.interaction
        reward_new_positions = self.state[temp_positions].interact(self.agents[idx])
        rewards, new_positions, is_terminal = reward_new_positions

        # Update new positions
        self.agent_positions = [pos for pos in self.agent_positions]
        self.agent_positions[idx] = new_positions

        # Return move results, in the same order as self.agents
        return rewards, self.extract_state(idx), self.goal.has_reached()

    # ----- Private Functions ----- #
    def process_action(
        self, action: Tuple[int, int], agent_position: List[Tuple[int, int]]
    ):
        # Move according to action
        x, y = agent_position
        dx, dy = action
        return x + dx, y + dy

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
        self.goal = goal
        used_pos.append(goal_pos)

        # Assign items to a random position in the remaining tiles
        item_pos = GridFactory.get_random_pos(self.width, self.height, used_pos)
        item = Item(item_pos)
        self.state[item_pos] = item
        self.lookup.add(item)
        used_pos.append(item_pos)
        self.item = item

        # Assign agents to random positions
        self.agent_positions = []
        for _ in self.agents:
            agent_pos = GridFactory.get_random_pos(self.width, self.height, used_pos)
            used_pos.append(agent_pos)
            self.agent_positions.append(agent_pos)

        self.max_reward = GridUtil.calculate_max_reward(self)

    # ----- Public Functions ----- #
    def reset(self):
        self.init_environment()
        self.set_interactive_tiles()

    def add_agent(self, agent: "Agent"):
        self.agents.append(agent)

    def get_size(self):
        return self.width, self.height

    def get_max_reward(self):
        return self.max_reward

    def get_items(self):
        return [x for x in self.lookup if isinstance(x, Item)]

    def get_untaken_item_pos(self):
        items = self.get_items()
        untaken_items = filter(lambda i: not i.taken, items)
        return [i.get_pos() for i in untaken_items]

    def item_taken(self):
        item = next((x for x in self.lookup if isinstance(x, Item)), [None])
        return item.taken

    def get_item_positions(self):
        return [item.get_pos() for item in self.get_items()]

    def get_goal_positions(self):
        goal = self.goal
        return goal.x, goal.y

    def extract_state(self, idx):
        x, y = self.agent_positions[idx]
        x2, y2 = self.get_item_positions()[0]
        x3, y3 = self.get_goal_positions()
        # print(x, y, x2, y2, x3, y3)
        # TODO: remove hardcoded item_pos indices
        # return agent_pos, item_pos[0], self.has_item()
        state = torch.zeros(state_size, dtype=dtype)
        state[x * side + y] = 1
        if not self.item_taken():
            state[side**2 + x2 * side + y2] = 1
        state[side**2 * 2 + x3 * side + y3] = 1

        return state

    def get_agent_positions(self):
        return self.agent_positions


class GridUtil:
    def calculate_max_reward(grid: Grid):
        # TODO: can only work with one agent and one item ATM
        x1, y1 = grid.get_agent_positions()[0]
        x2, y2 = grid.get_item_positions()[0]
        x3, y3 = grid.get_goal_positions()

        # Manhanttan distance from agent to obj and obj to goal
        dist_to_obj = abs(x1 - x2) + abs(y1 - y2)
        dist_to_goal = abs(x2 - x3) + abs(y2 - y3)

        # +100 for reward and +2 for 2 unneeded mark deduction when stepping on item and goal respectively
        return (dist_to_obj + dist_to_goal) * -1 + 102
