import itertools
import numpy as np

from typing import TYPE_CHECKING
from copy import deepcopy

from core import Item, Goal
from .action import Action

if TYPE_CHECKING:
    from core import Goal


class State:
    def __init__(self, agent_positions, lookup):
        self.agent_positions = agent_positions
        self.lookup = deepcopy(lookup)

    def get_possible_actions():
        # Generate possible actions
        return [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]

    # TODO: fix hardcode
    def get_possible_states(width, height):
        # Generate all possible states
        return 25 + 25 + 25 + 1
        positions = [(x, y) for x in range(width) for y in range(height)]
        has_items = [True, False]
        return itertools.product(positions, positions, has_items)

    # ----- Private Functions ----- #
    def get_goal(self) -> "Goal":
        return next((x for x in self.lookup if isinstance(x, Goal)), [None])

    def get_items(self):
        return [x for x in self.lookup if isinstance(x, Item)]

    def get_item_positions(self):
        return [item.get_pos() for item in self.get_items()]

    # TODO: fix hardcode
    def has_item(self):
        item = next((x for x in self.lookup if isinstance(x, Item)), [None])
        return item.taken

    def extract_state(self, idx):
        x, y = self.agent_positions[idx]
        x2, y2 = self.get_item_positions()[0]
        x3, y3 = self.get_goal_positions()
        has_item = self.has_item()
        # TODO: remove hardcoded item_pos indices
        # return agent_pos, item_pos[0], self.has_item()
        state = np.zeros(76)
        state[0] = 1 if has_item else 0
        state[1 + x * 5 + y] = 1
        state[26 + x2 * 5 + y2] = 1
        state[51 + x3 * 5 + y3] = 1
        return state

    # ----- Information Extraction ----- #
    def get_agent_positions(self):
        return self.agent_positions

    # TODO: fix hardcode
    def get_goal_positions(self):
        goal = self.get_goal()
        return goal.x, goal.y

    def get_item_positions(self):
        return [item.get_pos() for item in self.get_items()]

    # TODO: fix hardcode
    def is_terminal(self):
        goal = self.get_goal()
        return goal.has_reached()

    def get_untaken_item_pos(self):
        untaken_items = filter(lambda i: not i.taken, self.get_items())
        return [i.get_pos() for i in untaken_items]
