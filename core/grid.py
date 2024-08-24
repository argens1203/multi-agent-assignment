import random

from .state import State, Action
from .cell import *

class GridFactory:
    def get_random_pos(width, height, exclude = []):
        while True:
            position = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            if position not in exclude:
                return position

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
        
        return [(reward, self.get_state(), self.is_terminal()) for reward in rewards]

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
        for _ in self.agents:
            agent_pos = GridFactory.get_random_pos(self.width, self.height, [goal_pos] + used_pos)
            used_pos.append(agent_pos)
        self.agent_positions = used_pos

        for agent in self.agents:
            agent.update(State(self.agent_positions, self.lookup))

    def get_agent_positions(self):
        return self.agent_positions

    def get_goal_positions(self):
        goal = self.get_goal()
        return goal.x, goal.y

    def get_item_positions(self):
        return [item.get_pos() for item in self.get_items()]

    def get_state(self):
        state = State(self.agent_positions, self.lookup)
        return state

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
