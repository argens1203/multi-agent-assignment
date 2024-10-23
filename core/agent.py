from typing import TYPE_CHECKING, Tuple, List
from abc import abstractmethod, ABC

import numpy as np
import random
import torch

from constants import state_size, device, dtype, action_space

if TYPE_CHECKING:
    pass


class ExpBuffer:
    def __init__(self, max):
        self.max = max
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
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.is_terminals[indices],
        )

    def clear(self):
        self.__init__(self.max)

    def __len__(self):
        return len(self.states)


class Agent(ABC):
    def __init__(
        self,
        dqn,
        buffer,
        update_frequency=500,
        eps_decay=0.9999,
        eps_min=0.05,
        gamma=0.997,
        batch_size=200,
    ):
        self.dqn = dqn
        self.buffer = buffer
        self.actions = action_space

        # Agent Properties
        self.total_reward = 0
        self.step_count = 0
        self.have_secret = False

        # Initialize Q Table for all state-action to be 0
        self.batch_size = batch_size
        self.update_frequency = update_frequency  # Network update frequency

        # Initialize Learning param
        self.epsilon = 1
        self.decay_ratio = eps_decay
        self.epsilon_min = eps_min
        self.gamma = gamma

        self.learning = True

    # ----- Core Functions ----- #
    def choose_action(
        self, state: torch.tensor, choose_best: bool, episode_ratio: float
    ) -> Tuple[int, int]:
        if not choose_best and np.random.rand() < self.epsilon:
            self.epsilon_decay(episode_ratio)
            return random.choice(self.actions)
        else:
            # Extract immutable state information
            idx = torch.argmax(self.dqn.get_qvals(state))
            return self.actions[idx]

    def epsilon_decay(self, episode_ratio):
        self.epsilon *= self.decay_ratio
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def update(
        self,
        state: torch.tensor,
        action: Tuple[int, int],
        reward: int,
        next_state: torch.tensor,
        is_terminal: bool,
    ):
        self.total_reward += reward
        if not self.learning:
            return None, None

        self.buffer.insert(
            state, self.actions.index(action), reward, next_state, is_terminal
        )
        loss = None
        if len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states, is_terminals = self.buffer.extract(
                self.batch_size
            )
            rewards = rewards.to(device)
            targets = self.gamma * self.dqn.get_maxQ(next_states) + rewards

            # For terminal states, target_val is reward
            indices = is_terminals.nonzero().to(device)
            targets[indices] = rewards[indices]

            loss = self.dqn.train_one_step(states, actions, targets)

        if self.step_count >= self.update_frequency:
            self.dqn.update_target()
            self.step_count = 0
        else:
            self.step_count += 1

        return loss, self.epsilon

    # ----- Public Functions ----- #
    def have_secret_(self, new_value: bool):
        self.have_secret = new_value

    def reset(self):
        self.have_secret = False
        self.total_reward = 0

    def get_total_reward(self):
        return self.total_reward

    def enable_learning(self):
        # if not self.learning:
        # self.buffer.clear()
        self.learning = True

    def disable_learning(self):
        self.learning = False

    # ----- Private Functions ----- #
    @abstractmethod
    def get_type(self):
        pass

    def load(self, idx):
        self.dqn.load(idx)

    def save(self, idx):
        self.dqn.save(idx)

    def interact(self, other: "Agent"):
        rewarded = False
        if self.is_different_type(other):
            if not other.have_secret:
                rewarded = True
            self.have_secret = True
            other.have_secret_(True)
        return (50 if rewarded else 0), None, None

    def is_different_type(self, other: "Agent"):
        return other.get_type() != self.get_type()


class Agent1(Agent):
    def get_type(self):
        return 1


class Agent2(Agent):
    def get_type(self):
        return 2
