from typing import TYPE_CHECKING

import numpy as np
import random

if TYPE_CHECKING:
    from shared import State, Action


class Agent:
    def __init__(self, idx, all_states, actions):
        # Agent property (for illustration purposes)
        self.is_having_item = False
        self.total_reward = 0

        self.actions = actions  # TODO: encode different action for different state. How to initialize Q-Table
        self.idx = idx

        # Initialize Q Table for all state-action to be 0
        # TODO: use multi-D np array
        self.Q = np.zeros((all_states, len(actions)))

        # Initialize Learning param
        # TODO: fix resetting epsilon
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = -1
        self.gamma = 0.8
        self.alpha = 0.1

    # ----- Core Functions ----- #
    def choose_action(self, state: "State", explore=True):
        if explore and np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Extract immutable state information
            state_i = self.massage(state)
            return self.actions[np.argmax(self.Q[state_i])]

    def update_learn(
        self,
        state: "State",
        action: "Action",
        reward: int,
        next_state: "State",
        is_terminal: bool,
        learn=True,
    ):
        self.update(next_state, reward)

        # Extract immutable state information
        state_i = self.massage(state)
        nxt_state_i = self.massage(next_state)

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

    def get_total_reward(self):
        return self.total_reward

    def update(self, state: "State", reward=0):
        self.is_having_item = state.has_item()
        self.total_reward += reward

    def reset(self):
        self.is_having_item = False
        self.total_reward = 0

    def get_q_table(self):
        return self.Q

    # ----- Private Functions ----- #
    # Extract immutable information from State object
    def massage(self, state: "State"):
        return state.extract_state(self.idx)
