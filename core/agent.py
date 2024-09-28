from typing import TYPE_CHECKING, Tuple, List

import numpy as np
import random
import torch
from .given import *
from constants import state_size, device, dtype

if TYPE_CHECKING:
    from shared import State, Action


class ExpBuffer:
    def __init__(self):
        self.max = 1000
        self.itr = 0
        self.has_reached = False

        self.states = torch.empty((self.max, state_size), dtype=dtype)
        self.actions = torch.empty((self.max,), dtype=dtype)
        self.rewards = torch.empty((self.max,), dtype=dtype)
        self.next_states = torch.empty((self.max, state_size), dtype=dtype)
        self.is_terminals = torch.empty((self.max,), dtype=torch.bool)
        pass

    def insert(self, state, action, reward, next_state, is_terminal):
        self.itr %= self.max
        self.states[self.itr] = state
        self.actions[self.itr] = action
        self.rewards[self.itr] = reward
        self.next_states[self.itr] = next_state
        self.is_terminals[self.itr] = is_terminal
        self.itr += 1

        if self.itr >= self.max:
            self.has_reached = True

    def extract(self, batch_size) -> Tuple[List[List[int]], List[int], List[float]]:
        indices = np.random.randint(
            0, self.max if self.has_reached else self.itr, batch_size
        )
        # print(len(self.states))
        # print(indices)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.is_terminals[indices],
        )

    def __len__(self):
        return len(self.states)


class Agent:
    def __init__(self, idx, all_states, actions):
        # Agent property (for illustration purposes)
        self.is_having_item = False
        self.total_reward = 0

        self.actions = actions  # TODO: encode different action for different state. How to initialize Q-Table
        self.idx = idx

        # Initialize Q Table for all state-action to be 0
        # TODO: use multi-D np array
        self.min_buffer = 200
        self.step_count = 0
        self.C = 500
        self.Q = np.zeros((all_states, len(actions)))

        # Initialize Learning param
        # TODO: fix resetting epsilon
        self.epsilon = 1.0
        self.epsilon_decay = 0.997  # TODO: reduce the decay (ie. increase the number)
        self.epsilon_min = 0.1
        self.gamma = 0.997

        # self.alpha = 0.1

        self.buffer = ExpBuffer()

        prepare_torch()

    # ----- Core Functions ----- #
    def choose_action(self, state: "State", explore=True):
        if explore and np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Extract immutable state information
            state_i = self.massage(state)
            idx = torch.argmax(get_qvals(state_i))
            return self.actions[idx]

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
        # nxt_state_i = self.massage(next_state)

        if not learn:
            return

        # # All states (including terminal states) have initial Q-values of 0 and thus there is no need for branching for handling terminal next state
        # self.Q[state_i][self.actions.index(action)] += self.alpha * (
        #     reward
        #     + self.gamma * np.max(self.Q[nxt_state_i])
        #     - self.Q[state_i][self.actions.index(action)]
        # )

        state_i = self.massage(state)
        nxt_state_i = self.massage(next_state)
        # target_val = self.gamma * get_maxQ(nxt_state_i) + reward
        # if is_terminal:
        #     target_val = torch.tensor(reward)

        # next_qa = np.copy(current_qa)
        # next_qa[np.argmax(next_qa)] = target_val
        self.buffer.insert(
            state_i, self.actions.index(action), reward, nxt_state_i, is_terminal
        )
        if len(self.buffer) >= self.min_buffer:
            states, actions, rewards, next_states, is_terminals = self.buffer.extract(
                200
            )
            rewards = rewards.to(device)
            indices = is_terminals.nonzero().to(device)
            targets = self.gamma * get_maxQ(next_states.to(device)) + rewards
            targets[indices] = rewards[
                indices
            ]  # For terminal states, target_val is reward
            # print(states, actions, targets)
            # print(states.shape, actions.shape, targets.shape)
            loss = train_one_step(states, actions, targets)

        if self.step_count >= self.C:
            update_target()
            self.step_count = 0
        else:
            self.step_count += 1

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

    # ----- Public Functions ----- #
    def has_item(self):
        return self.is_having_item

    def get_total_reward(self):
        return self.total_reward

    def set_has_item(self, has_item: bool):
        self.is_having_item = has_item

    def update(self, state: "State", reward=0):
        self.total_reward += reward

    def reset(self):
        self.is_having_item = False
        self.total_reward = 0

    def get_q_table(self):
        return self.Q

    # ----- Private Functions ----- #
    # Extract immutable information from State object
    def massage(self, state: "State"):
        state_i = state.extract_state(self.idx).to(device).float()
        return state_i + 0.001  # 5 Minutes
        # return state_i + torch.rand(state_size).to(device) / 100.0  # 15 Minutes
