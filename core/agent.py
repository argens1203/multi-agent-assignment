import numpy as np
import random

class Agent:
    def __init__(self, all_states, actions):
        # Agent property (for illustration purposes)
        self.position = None
        self.has_item = False

        # Initialize action vector
        self.actions = actions # TODO: encode different action per state. How to initialize Q-Table
        
        # Initialize Q Table for all state-action to be 0
        self.Q = {}
        for state in all_states:
            self.Q[state] = np.zeros(
                len(actions)
            )

        # Initialize Learning param
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = -1
        self.gamma = 0.8
        self.alpha = 0.1

        # Initialize Agent metrics
        self.total_reward = 0

    def choose_action(self, state, explore = True):
        if explore and np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.Q[state])]

    def perceive(self, state, action, reward, next_state, is_terminal, learn = True):
        # Update internal params wrt to updated state
        self.set_state(state)
        if not learn:
            return

        # All states (including terminal states) have initial Q-values of 0 and thus there is no need for branching for handling terminal next state
        self.Q[state][self.actions.index(action)] += self.alpha * (
            reward
            + self.gamma * np.max(self.Q[next_state])
            - self.Q[state][self.actions.index(action)]
        )

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_state(self, state):
        agent_pos, item_pos, has_item = state

        self.position = agent_pos
        self.has_item = has_item

    def reset(self):
        pass
