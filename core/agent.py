import numpy as np
import random

class Agent:
    def __init__(self, idx, all_states, actions): # TODO: store idx for multi-agent scenario
        # Agent property (for illustration purposes)
        self.is_having_item = False

        # Initialize action vector
        self.actions = actions # TODO: encode different action per state. How to initialize Q-Table
        self.idx = idx
        
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

    # ----- Core Functions ----- #
    def choose_action(self, state, explore = True):
        state_i = self.massage(state)
        if explore and np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.Q[state_i])]

    def update_learn(self, state, action, reward, next_state, is_terminal, learn = True):
        self.update(next_state)

        state_i = self.massage(state)
        nxt_state_i = self.massage(next_state)

        # Update internal params wrt to updated state
        if not learn:
            return

        # All states (including terminal states) have initial Q-values of 0 and thus there is no need for branching for handling terminal next state
        self.Q[state_i][self.actions.index(action)] += self.alpha * (
            reward
            + self.gamma * np.max(self.Q[nxt_state_i])
            - self.Q[state_i][self.actions.index(action)]
        )

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ----- Public Functions ----- #
    def has_item(self):
        return self.is_having_item

    def update(self, state):
        self.is_having_item = state.has_item()

    def reset(self):
        pass

    def massage(self, state):
        return state.extract_state(self.idx)

    # ----- Private Functions ----- #