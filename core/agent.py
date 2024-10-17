from typing import TYPE_CHECKING, Tuple, List
from abc import abstractmethod, ABC

import numpy as np
import random
import torch
from .given import DQN
from constants import state_size, device, dtype, action_size, debug

if TYPE_CHECKING:
    pass


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


buffer1 = ExpBuffer()
buffer2 = ExpBuffer()
dqn1 = DQN(state_size=state_size, action_size=action_size)
dqn2 = DQN(state_size=state_size, action_size=action_size)


class Agent(ABC):
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
        self.epsilon = 0.2
        self.epsilon_decay = 0.997  # TODO: reduce the decay (ie. increase the number)
        self.epsilon_min = 0.01
        self.gamma = 0.997

    # ----- Core Functions ----- #
    def choose_action(
        self, state: torch.tensor, choose_best: bool, ep_ratio: float
    ) -> Tuple[int, int]:
        eps = 1 - (1 - self.epsilon_min) * min(1, ep_ratio)
        if not choose_best and np.random.rand() < eps:
            return random.choice(self.actions)
        else:
            # Extract immutable state information
            state_i = self.massage(state)
            idx = torch.argmax(self.dqn.get_qvals(state_i))
            return self.actions[idx]

    def update(
        self,
        state: torch.tensor,
        action: Tuple[int, int],
        reward: int,
        next_state: torch.tensor,
        is_terminal: bool,
        learn: bool,
    ):
        self.total_reward += reward
        if not learn:
            return

        state_i = self.massage(state)
        nxt_state_i = self.massage(next_state)
        self.buffer.insert(
            state_i, self.actions.index(action), reward, nxt_state_i, is_terminal
        )
        if len(self.buffer) >= self.min_buffer:
            states, actions, rewards, next_states, is_terminals = self.buffer.extract(
                200
            )
            rewards = rewards.to(device)
            indices = is_terminals.nonzero().to(device)
            targets = self.gamma * self.dqn.get_maxQ(next_states.to(device)) + rewards
            targets[indices] = rewards[
                indices
            ]  # For terminal states, target_val is reward
            loss = self.dqn.train_one_step(states, actions, targets)

        if self.step_count >= self.C:
            self.dqn.update_target()
            self.step_count = 0
        else:
            self.step_count += 1

        return loss

    # ----- Public Functions ----- #
    def have_secret_(self, new_value: bool):
        self.is_have_secret = new_value

    def has_secret(self):
        return self.is_have_secret

    def reset(self):
        self.is_having_item = False
        self.is_have_secret = False
        self.total_reward = 0

    @abstractmethod
    def get_type(self):
        pass

    def get_total_reward(self):
        return self.total_reward

    # ----- Private Functions ----- #
    # Extract immutable information from State object
    def massage(self, state: torch.tensor):
        state_i = state.to(device).float()
        return state_i + 0.001  # 5 Minutes
        # return state_i + torch.rand(state_size).to(device) / 100.0  # 15 Minutes

    def interact(self, other: "Agent"):
        return 0, None, None


class Agent1(Agent):
    def __init__(self, idx, all_states, actions):
        super().__init__(idx, all_states, actions)
        self.buffer = buffer1
        self.dqn = dqn1
        self.is_have_secret = False

    def get_type(self):
        return 1

    def interact(self, other: "Agent"):
        rewarded = False
        if other.get_type() == 2:
            if not other.has_secret():
                rewarded = True
            self.is_have_secret = True
            other.have_secret_(True)
        return (50 if rewarded else 0), None, None

    def reset(self):
        super().reset()
        self.is_have_secret = False


class Agent2(Agent):
    def __init__(self, idx, all_states, actions):
        super().__init__(idx, all_states, actions)
        self.buffer = buffer2
        self.dqn = dqn2
        self.is_have_secret = False

    def get_type(self):
        return 2

    def interact(self, other: "Agent"):
        rewarded = False
        if other.get_type() == 1:
            if not other.has_secret():
                rewarded = True
            self.is_have_secret = True
            other.have_secret_(True)
        return (50 if rewarded else 0), None, None

    def reset(self):
        super().reset()
        self.is_have_secret = False
