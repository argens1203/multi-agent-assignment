import numpy as np
import random
import itertools
class Agent:
    def __init__(self, env):
        self.env = env
        self.actions = ['north', 'south', 'east', 'west']
        # Define the grid size (n x n)
        grid_size = self.env.size
        # Initialize the dictionary
        q_table = {}
        # Generate all possible positions within the grid
        all_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]

        # Generate all possible values of has_item (True/False)
        has_item_values = [True, False]

        # Iterate over all combinations of agent_position, item_position, and has_item
        for agent_position, item_position, has_item in itertools.product(all_positions, all_positions, has_item_values):
            # Initialize the Q-value for this state as 0
            q_table[(agent_position, item_position, has_item)] = np.zeros(len(self.actions))

        self.Q = q_table
        # self.q_table = np.zeros(   (self.env.size**2, self.env.size**2, 2, len(self.actions))   )
        self.epsilon = 0.2
        self.gamma = 0.9
        self.alpha = 0.1
    def get_position(self):
        return self.env.get_state()[0]
  
    def has_item(self):
        return self.env.get_state()[2]
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            # print(state)
            q_values = self.Q[state]
            return self.actions[np.argmax(q_values)]
    
    def learn(self, state, action, reward, next_state):
        self.Q[state][self.actions.index(action)] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][self.actions.index(action)])
    

    def move(self):
        state = self.env.get_state()
        action = self.choose_action(state)
        print(action)
        next_state, reward, terminal = self.env.move(action)

    def train(self, episodes=1000):
        for _ in range(episodes):
            self.env.reset()
            state = self.env.get_state()
            total_reward = 0
            while not self.env.is_terminal():
                action = self.choose_action(state)
                next_state, reward, terminal = self.env.move(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                # print(f"Episode {_} completed with total reward: {total_reward}")

