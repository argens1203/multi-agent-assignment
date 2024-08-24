import itertools
from copy import deepcopy

from .cell import Item, Goal


class Action:
    NORTH = "N"
    WEST = "W"
    EAST = "E"
    SOUTH = "S"


class State:
    def __init__(self, agent_positions, lookup):
        self.agent_positions = agent_positions
        self.lookup = deepcopy(lookup)

    def get_possible_actions():
        # Generate possible actions
        return [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]

    def get_possible_states(width, height):
        # Generate all possible states
        positions = [(x, y) for x in range(width) for y in range(height)]
        has_items = [True, False]
        return itertools.product(positions, positions, has_items)

    # ----- Private Functions ----- #
    def get_goal(self):
        return next((x for x in self.lookup if isinstance(x, Goal)), [None])

    def get_items(self):
        return [x for x in self.lookup if isinstance(x, Item)]

    def get_item_positions(self):
        return [item.get_pos() for item in self.get_items()]

    def has_item(self):
        item = next((x for x in self.lookup if isinstance(x, Item)), [None])
        return item.taken

    def extract_state(self, idx):
        item_pos = self.get_item_positions()[idx]
        agent_pos = self.agent_positions[idx]
        return agent_pos, item_pos, self.has_item()

    # ----- Information Extraction ----- #
    def get_agent_positions(self):
        return self.agent_positions

    def get_goal_positions(self):
        goal = self.get_goal()
        return goal.x, goal.y

    def get_item_positions(self):
        return [item.get_pos() for item in self.get_items()]

    def is_terminal(self):
        goal = self.get_goal()
        return goal.has_reached()

    def get_untaken_item_pos(self):
        untaken_items = filter(lambda i: not i.taken, self.get_items())
        return [i.get_pos() for i in untaken_items]
