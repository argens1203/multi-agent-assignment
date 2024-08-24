import numpy as np
import random
from .agent import Agent

class Action:
    NORTH = "N"
    WEST = "W"
    EAST = "E"
    SOUTH = "S"

class GridFactory:
    def get_random_pos(width, height, exclude = []):
        while True:
            position = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            if position not in exclude:
                return position

class Empty:
    def __init__(self, pos):
        x, y = pos
        self.x = x
        self.y = y

    def interact(self, other: Agent):
        return -1, (self.x, self.y)

class Goal(Empty):
    def __init__(self, pos):
        x, y = pos
        self.x = x
        self.y = y
        self.reached = False

    def interact(self, other: Agent):
        has_item = other.has_item()
        if has_item:
            self.reached = True
            return 50, (self.x, self.y)
        else:
            return -1, (self.x, self.y)
    
    def has_reached(self):
        # print('self.has_reached', self.reached)
        return self.reached

class Item(Empty):
    def __init__(self, pos):
        self.taken = False
        x, y = pos
        self.x = x
        self.y = y
    
    def interact(self, other: Agent):
        # print('interacting')
        if not self.taken:
            self.taken = True
            # print('taken')
            return 50, (self.x, self.y)
        
        return -1, (self.x, self.y)

    def get_pos(self):
        return self.x, self.y

class Wall(Empty):
    def __init__(self, pos, dimensions):
        x, y = pos
        self.x = x
        self.y = y

        width, height = dimensions
        self.new_x = min(width - 1, max(0, x))
        self.new_y = min(height - 1, max(0, y))

    def interact(self, other: Agent):
        return -10, (self.new_x, self.new_y)

class GridWorld:
    def __init__(self, width = 5, height = 5):
        self.width = width
        self.height = height

        self.state = {} # TODO: multiple entities in one cell
        self.lookup = set()
        self.agents = []
        self.agent_positions = []

        for x in range(-1, width + 1):
            for y in range(-1, height + 1):
                if x < 0 or x >= width:
                    self.state[(x, y)] = Wall((x, y), (width, height))
                elif y < 0 or y >= height:
                    self.state[(x, y)] = Wall((x, y), (width, height))
                else:
                    self.state[(x, y)] = Empty((x, y))

    # ----- Core Functions ----- #
    def move(self, actions):
        temp_positions = [self.process_action(action, agent_pos) for action, agent_pos in zip(actions, self.agent_positions)]
        reward_new_positions = [self.state[(x, y)].interact(agent) for agent, (x, y) in zip(self.agents, temp_positions)]
        rewards, new_positions = zip(*reward_new_positions)
        
        self.agent_positions = new_positions

        reward_new_states = [(reward, (new_pos, self.get_item_positions()[0], self.has_item())) for reward, new_pos in reward_new_positions]
        return [(reward, new_state, self.is_terminal()) for reward, new_state in reward_new_states]

    def process_action(self, action, agent_position):
        # Move according to action
        x, y = agent_position
        dx, dy = self.interpret_action(action)
        return x + dx, y + dy

    # ----- Private Functions ----- #
    def interpret_action(self, action):
        if action == Action.NORTH:
            return 0, -1
        if action == Action.SOUTH:
            return 0, 1
        if action == Action.EAST:
            return 1, 0
        if action == Action.WEST:
            return -1, 0

    # ----- Private Functions ----- #
    def get_items(self):
        return [x for x in self.lookup if isinstance(x, Item)]
        # return next((x for x in self.lookup if isinstance(x, Item)), [None])

    def get_goal(self):
        return next((x for x in self.lookup if isinstance(x, Goal)), [None])

    def has_item(self):
        item = next((x for x in self.lookup if isinstance(x, Item)), [None])
        return item.taken

    def update_agent_positions(self, new_position):
        agent_position, item_position, has_item = self.get_state()
        return new_position, item_position, has_item

    # ----- Public Functions ----- #
    def add_agents(self, agents):
        self.agents = agents

    def reset(self):
        self.lookup.clear()

        goal_pos = (self.width - 1, self.height - 1)
        goal = Goal(goal_pos)
        self.state[goal_pos] = goal
        self.lookup.add(goal)

        item_pos = GridFactory.get_random_pos(self.width, self.height, [goal_pos])
        item = Item(item_pos)
        self.state[item_pos] = item
        self.lookup.add(item)

        used_pos = []
        for agent in self.agents:
            agent_pos = GridFactory.get_random_pos(self.width, self.height, [goal_pos] + used_pos)
            agent.update((agent_pos, item_pos, False))
            used_pos.append(agent_pos)
        self.agent_positions = used_pos

    def get_agent_positions(self):
        return self.agent_positions

    def get_goal_positions(self):
        goal = self.get_goal()
        return goal.x, goal.y

    def get_item_positions(self):
        return [item.get_pos() for item in self.get_items()]

    def get_state(self):
        return (self.get_agent_positions()[0], self.get_item_positions()[0], self.has_item()) # TODO: for multiple agent and items

    def is_terminal(self):
        goal = self.get_goal()
        return goal.has_reached()
    
    def get_untaken_item_pos(self):
        untaken_items = filter(lambda i: not i.taken, self.get_items())
        return [i.get_pos() for i in untaken_items]

    def get_size(self):
        return self.width, self.height

class GridUtil:
    def calculate_max_reward(grid):
        x1, y1 = grid.get_agent_positions()[0]
        x2, y2 = grid.get_item_positions()[0]
        x3, y3 = grid.get_goal_positions()
        dist_to_obj = abs(x1 - x2) + abs(y1 - y2)
        dist_to_goal = abs(x2 - x3) + abs(y2 - y3)

        return (dist_to_obj + dist_to_goal) * -1 + 102    
