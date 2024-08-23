import numpy as np
import random
import itertools


class Agent:
    def __init__(self, env):
        self.env = env
        self.actions = ["north", "south", "east", "west"]
        # Define the grid size (n x n)
        grid_size = self.env.size
        # Initialize the dictionary
        q_table = {}
        # Generate all possible positions within the grid
        all_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]

        # Generate all possible values of has_item (True/False)
        has_item_values = [True, False]

        # Iterate over all combinations of agent_position, item_position, and has_item
        for agent_position, item_position, has_item in itertools.product(
            all_positions, all_positions, has_item_values
        ):
            # Initialize the Q-value for this state as 0
            q_table[(agent_position, item_position, has_item)] = np.zeros(
                len(self.actions)
            )

        self.Q = q_table
        # self.q_table = np.zeros(   (self.env.size**2, self.env.size**2, 2, len(self.actions))   )
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = -1
        self.gamma = 0.8
        self.alpha = 0.1
        self.total_reward = 0

        self.position = None
        self.has_item = False

    def set_state(self, state):
        agent_pos, item_pos, has_item = state
        self.position = agent_pos
        self.has_item = has_item

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:  # TODO: epsilon should decay
            return random.choice(self.actions)
        else:
            # print(state)
            q_values = self.Q[state]
            return self.actions[np.argmax(q_values)]

    def perceive(self, state, action, reward, next_state, is_terminal):
        # Update internal params wrt to updated state
        self.set_state(state)

        # All states (including terminal states) have initial Q-values of 0 and thus there is no need for branching for handling terminal next state
        self.Q[state][self.actions.index(action)] += self.alpha * (
            reward
            + self.gamma * np.max(self.Q[next_state])
            - self.Q[state][self.actions.index(action)]
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def move(self):
        if self.env.is_terminal():
            return
        state = self.env.get_state()
        action = self.choose_action(
            state
            # TODO: disable epsilon whent testing
        )  # TODO: Should only choose among valid moves?
        print(action)
        next_state, reward, terminal = self.env.move(action)
        self.set_state(next_state)
        self.total_reward += reward

    def reset(self):
        pass
