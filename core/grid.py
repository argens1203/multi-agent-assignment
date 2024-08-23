import numpy as np
import random


class Action:
    NORTH = "N"
    WEST = "W"
    EAST = "E"
    SOUTH = "S"

class GridFactory:
    def get_random_pos(size, exclude = []):
        while True:
            position = (
                random.randint(0, size - 1),
                random.randint(0, size - 1),
            )
            if position not in exclude:
                return position

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.goal_position = (self.size - 1, self.size - 1)
        self.setup()

    # ----- Core Functions ----- #
    def move(self, action):
        # Move according to action
        x, y = self.agent_position
        dx, dy = self.interpret_action(action)
        new_x, new_y = x + dx, y + dy
        
        # Calculate reward according to new position
        reward = self.calculate_reward(new_x, new_y)
        
        # Clip new position to within grid
        new_x = min(self.size - 1, max(0, new_x))
        new_y = min(self.size - 1, max(0, new_y))

        # Update cached position
        if (x, y) == self.item_position and not self._has_item:
            self._has_item = True
        self.agent_position = (new_x, new_y)

        return self.get_state(), reward, self.is_terminal()

    # ----- Private Functions ----- #
    def setup(self):
        self.agent_position = GridFactory.get_random_pos(self.size, [self.goal_position])
        self.item_position = GridFactory.get_random_pos(self.size, [self.agent_position, self.goal_position])
        self._has_item = False

    def calculate_reward(self, x, y):
        # Going out of bounds
        if x < 0 or x >= self.size:
            return -10
        if y < 0 or y >= self.size:
            return -10

        # Going to item before getting item
        if (x, y) == self.item_position and not self._has_item:
            return 50

        # Going to goal with item
        if (x, y) == self.goal_position and self._has_item:
            return 50

        return -1

    def interpret_action(self, action):
        if action == Action.NORTH:
            return 0, -1
        if action == Action.SOUTH:
            return 0, 1
        if action == Action.EAST:
            return 1, 0
        if action == Action.WEST:
            return -1, 0

    # ----- Public Functions ----- #
    def reset(self):
        self.setup()

    def get_goal_positions(self):
        return self.goal_position

    def get_item_positions(self):
        return [self.item_position]

    def get_state(self):
        return (self.agent_position, self.item_position, self._has_item)

    def is_terminal(self):
        return self.agent_position == self.goal_position and self._has_item

class GridUtil:
    def calculate_max_reward(grid):
        x1, y1 = grid.agent_position
        x2, y2 = grid.item_position
        x3, y3 = grid.goal_position
        dist_to_obj = abs(x1 - x2) + abs(y1 - y2)
        dist_to_goal = abs(x2 - x3) + abs(y2 - y3)

        return (dist_to_obj + dist_to_goal) * -1 + 102    
