import numpy as np
import random
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.B_position = (self.size - 1, self.size - 1)
        self.agent_position = self.random_position(exclude=[self.B_position])
        self.item_position = self.random_position(exclude=[self.agent_position, self.B_position])
        
        self.has_item = False
    
    def random_position(self, exclude=[]):
        # return (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        while True:
                position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if position not in exclude:
                    return position
    
    def reset(self):
        self.agent_position = self.random_position(exclude=[self.B_position])
        self.item_position = self.random_position(exclude=[self.agent_position, self.B_position])
        self.has_item = False

    def reward(self):
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
        if action == 'north' and y > 0:
            y -= 1
        elif action == 'south' and y < self.size - 1:
            y += 1
        elif action == 'west' and x > 0:
            x -= 1
        elif action == 'east' and x < self.size - 1:
            x += 1
        if (x, y) == self.agent_position:
            reward = -10
        else:
            reward = self.reward()
        self.agent_position = (x, y)
        #We update "has_item" in the reward function
        # if self.agent_position == self.item_position:
        #     self.has_item = True
        return self.get_state(), reward, self.is_terminal()


