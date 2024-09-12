import random

from shared import State, Action
from .cell import *


class GridFactory:
    # Getting a random location in a grid, excluding certain locations
    def get_random_pos(width, height, exclude=[]):
        while True:
            position = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            if position not in exclude:
                return position


class Grid:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

        self.state = {}  # TODO: multiple entities in one cell
        self.lookup = set()  # Interactive tiles
        self.agents = []
        self.agent_positions = []

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
                    self.state[(x, y)] = Empty((x, y))

    # ----- Core Functions ----- #
    def move(self, actions):  # List of actions, in the same order as self.agents
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
        rewards, new_positions = zip(*reward_new_positions)

        # Update new positions
        self.agent_positions = new_positions

        # Return move results, in the same order as self.agents
        return [
            (reward, self.get_state(), self.get_state().is_terminal())
            for reward in rewards
        ]

    # ----- Private Functions ----- #
    def process_action(self, action, agent_position):
        # Move according to action
        x, y = agent_position
        dx, dy = self.interpret_action(action)
        return x + dx, y + dy

    def interpret_action(self, action):
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
        goal_pos = (self.width - 1, self.height - 1)
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

    # ----- Public Functions ----- #
    def reset(self):
        self.init_environment()
        self.set_interactive_tiles()

    def add_agents(self, agents):
        self.agents = agents

    def get_state(self):
        return State(self.agent_positions, self.lookup)


class GridUtil:
    def calculate_max_reward(grid):
        # TODO: can only work with one agent and one item ATM
        x1, y1 = grid.get_state().get_agent_positions()[0]
        x2, y2 = grid.get_state().get_item_positions()[0]
        x3, y3 = grid.get_state().get_goal_positions()

        # Manhanttan distance from agent to obj and obj to goal
        dist_to_obj = abs(x1 - x2) + abs(y1 - y2)
        dist_to_goal = abs(x2 - x3) + abs(y2 - y3)

        # +100 for reward and +2 for 2 unneeded mark deduction when stepping on item and goal respectively
        return (dist_to_obj + dist_to_goal) * -1 + 102
