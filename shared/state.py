import itertools

from copy import deepcopy

from core import Item, Goal
from .action import Action


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
        return 5**5
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

    # TODO: fix hardcode
    def has_item(self):
        item = next((x for x in self.lookup if isinstance(x, Item)), [None])
        return item.taken

    def extract_state(self, idx):
        x, y = self.agent_positions[idx]
        x2, y2 = self.get_item_positions()[0]
        # TODO: remove hardcoded item_pos indices
        # return agent_pos, item_pos[0], self.has_item()
        return (
            x * (5**4)
            + y * (5**3)
            + x2 * (5**2)
            + y2 * (5)
            + (1 if self.has_item() else 0)
        )

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
