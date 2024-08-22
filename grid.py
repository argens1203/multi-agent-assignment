import numpy as np
import random


class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.B_position = (self.size - 1, self.size - 1)
        self.agent_position = self.random_position(exclude=[self.B_position])
        self.item_position = self.random_position(
            exclude=[self.agent_position, self.B_position]
        )
        self.has_item = False
        self.max_reward = self.calculate_max_reward()

    def random_position(self, exclude=[]):
        # return (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        while True:
            position = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )
            if position not in exclude:
                return position

    def reset(self):
        self.agent_position = self.random_position(exclude=[self.B_position])
        self.item_position = self.random_position(
            exclude=[self.agent_position, self.B_position]
        )
        self.has_item = False

    def calculate_max_reward(self):
        x1, y1 = self.agent_position
        x2, y2 = self.item_position
        x3, y3 = self.B_position
        dist_to_obj = abs(x1 - x2) + abs(y1 - y2)
        dist_to_goal = abs(x2 - x3) + abs(y2 - y3)

        return (dist_to_obj + dist_to_goal) * -1 + 102

    def reward(self):  # TODO: Can increase penalty for hitting the wall
        if self.agent_position == self.item_position and not self.has_item:
            self.has_item = True
            return 50
        if self.agent_position == self.B_position and self.has_item:
            return 50
        return -1

    def get_state(self):
        return (self.agent_position, self.item_position, self.has_item)

    def is_terminal(self):
        return self.agent_position == self.B_position and self.has_item

    def move(self, action):
        # Define movement logic for 'north', 'south', 'east', 'west'
        x, y = self.agent_position
        if action == "north" and y > 0:
            y -= 1
        elif action == "south" and y < self.size - 1:
            y += 1
        elif action == "west" and x > 0:
            x -= 1
        elif action == "east" and x < self.size - 1:
            x += 1
        if (x, y) == self.agent_position:
            reward = -10
        else:
            self.agent_position = (x, y)
            reward = self.reward()

        # We update "has_item" in the reward function
        # if self.agent_position == self.item_position:
        #     self.has_item = True
        return self.get_state(), reward, self.is_terminal()
