import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Dict, Set

import random

from constants import dtype, state_size, side, debug
import datetime
from .view import IVisual

if TYPE_CHECKING:
    from .agent import Agent
    from .storage import Storage


class Controller:
    # Iterate by number of games
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_reset = True

    def toggle_auto_reset(self):
        self.auto_reset = not self.auto_reset
        return self.auto_reset

    def next(self):
        if self.has_ended() and self.auto_reset:
            self.reset()
        self.step(is_testing=True)
        return


class Trainer:
    def __init__(self, storage, **kwargs):
        super().__init__(**kwargs)
        self.storage = storage
        self.learners = [1, 2]

    def train(self, itr=1):
        start = datetime.datetime.now()
        print(f"Start Time: {start}")
        self.reset()
        for i in range(itr):
            # self.enable_learning(agent_type=1)
            # self.enable_learning(agent_type=2)
            if i / itr >= 0.9:
                self.learners = [1, 2]
            else:
                if (i // (itr // 1000)) % 2 == 0:
                    self.learners = [1]
                    self.enable_learning(agent_type=1)
                    self.disable_learning(agent_type=2)
                else:
                    self.learners = [2]
                    self.enable_learning(agent_type=2)
                    self.disable_learning(agent_type=1)

                    # self.enable_learning(agent_type=(2 - (i // (itr // 100)) % 2))
                    # self.disable_learning(agent_type=(1 + (i // (itr // 100)) % 2))

            self.reset()
            (loss, reward, epsilon, ml_losses, step_count) = self.play_one_game(
                ep=i, total_ep=itr
            )

            self.storage.append_ml_losses(ml_losses)
            self.storage.append_loss_epsilon(loss, epsilon)

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch: {i+1}/{itr} -- Time Elapsed: {datetime.datetime.now() - start}"
                )
        return self.storage.ml_losses

    def small_test(self, itr=1):
        self.reset()
        total_loss = 0
        total_step_count = 0
        success_count = 0
        for agent in self.agents:
            agent.disable_learning()
        for i in range(itr):
            self.reset()
            loss, reward, eps, ml_losses, step_count = self.play_one_game(
                is_testing=True
            )
            if step_count < 15:
                success_count += 1
            total_loss += loss
            total_step_count += step_count
            if (i + 1) % 100 == 0:
                print(
                    f"Ep {i + 1}: Avg = {total_step_count / (i + 1)}; Excess = {total_loss / (i + 1)}; Success = {success_count/(i + 1)}"
                )
        return total_loss / itr

    def test(self, agents=[], itr=1):
        self.storage.reset_test_loss()
        self.learners = []
        for agent in self.agents:
            agent.disable_learning()

        possible_loc = [(x, y) for x in range(side) for y in range(side)]
        total_step_count = 0
        excess_step_count = 0
        success_count = 0
        count = 0
        for goal_pos in possible_loc:
            for p1 in possible_loc:
                for p2 in possible_loc:
                    for p3 in possible_loc:
                        for p4 in possible_loc:
                            self.step_count = 0
                            self.goal_reached = False
                            self.itr = 0

                            self.agents = agents
                            self.agent_positions = [p1, p2, p3, p4]
                            self.goal_pos = goal_pos
                            self.min_step = self.calculate_min_step()
                            for agent in self.agents:
                                agent.reset()
                            self.reorder_for_min_step()

                            (loss, reward, epsilon, ml_loss, step_count) = (
                                self.play_one_game(is_testing=True)
                            )
                            self.storage.append_test_loss(loss)
                            self.storage.append_step_count(step_count)

                            total_step_count += step_count
                            excess_step_count += step_count - self.min_step
                            # print(excess_step_count, step_count, self.min_step)
                            # print(p1, p2, p3, p4, goal_pos)
                            # input()
                            if step_count < 15:
                                success_count += 1
                            count += 1

                            if count % 100 == 0:
                                print(
                                    f"Ep {count}: Avg = {total_step_count / count}; Excess = {excess_step_count / count}; Success = {success_count/count}"
                                )
        return self.storage.test_loss, self.storage.step_count

    def enable_learning(self, agent_type):
        agents = [a for a in self.agents if a.get_type() == agent_type]
        for a in agents:
            a.enable_learning()

    def disable_learning(self, agent_type):
        agents = [a for a in self.agents if a.get_type() == agent_type]
        for a in agents:
            a.disable_learning()

    def play_one_game(self, **kwargs):
        # self.reset()

        max_step_count = 50
        self.step_count = 0
        ml_losses = []
        epsilons = []
        while not self.has_ended() and self.step_count < max_step_count:
            ml_loss, epsilon = self.step(**kwargs)
            if ml_loss is not None:
                ml_losses.append(ml_loss)
            if epsilon is not None:
                epsilons.append(epsilon)
            self.step_count += 1
        # print(self.step_count)
        total_reward = sum(map(lambda a: a.get_total_reward(), self.agents))
        loss = self.step_count - self.min_step
        return (
            loss,
            total_reward,
            None if len(epsilons) == 0 else epsilons[-1],
            ml_losses,
            self.step_count,
        )  # TODO: 0

    def train_in_background(self):
        gp, tp, conn1 = get_process(self.storage, self)
        gp.start()
        tp.start()
        gp.join()
        tp.join()

        name = conn1.recv()
        # TODO: return array of trained_Q
        trained_Q = get_np_from_name(name)
        return trained_Q

    def test_in_background(self, ep=1000):
        gp, tp = get_test_process(self.storage, self, ep)
        gp.start()
        tp.start()
        gp.join()
        tp.join()

    def try_load_dqn(self):
        for idx, agent in enumerate(self.agents):
            try:
                agent.load(idx)
            except Exception as e:
                print(e)
                print("load failed")
                pass

    def save_dqn(self):
        for idx, agent in enumerate(self.agents):
            agent.save(idx)


class Visual:
    def get_agent_info(self) -> List[Tuple[Tuple[int, int], int, bool]]:
        """
        Output: List of
                - Tuple of:
                    - coordinate: (int, int)
                    - type: int (1 or 2)
                    - have_secret: bool
        """
        agent_types = map(lambda agent: agent.get_type(), self.agents)
        have_secrets = map(lambda agent: agent.have_secret, self.agents)
        return list(zip(self.get_agent_positions(), agent_types, have_secrets))

    def get_total_reward(self):
        return sum(map(lambda a: a.get_total_reward(), self.agents))

    def get_min_step(self):
        return self.min_step

    def get_size(self):
        return self.width, self.height

    def get_goal_positions(self):
        return self.goal_pos

    def get_agent_positions(self):
        return self.agent_positions

    def has_ended(self) -> bool:
        return self.goal_reached

    def get_step_count(self) -> int:
        return self.step_count


class GridUtil:
    def calculate_min_step_for_two(self, one_pos, two_pos, goal_pos, clock=False):
        one_dist = self.calc_mht_dist(one_pos, goal_pos)
        two_dist = self.calc_mht_dist(two_pos, goal_pos)

        smaller, larger = min(one_dist, two_dist), max(one_dist, two_dist)
        smaller = smaller + (
            0
            if self.line_passing_through_goal_can_cut(one_pos, two_pos, goal_pos)
            else 2
        )
        smaller, larger = min(smaller, larger), max(smaller, larger)

        return max(smaller, larger - (1 if clock else 0))

    def line_passing_through_goal_can_cut(self, one_pos, two_pos, goal_pos):
        x1, y1 = one_pos
        x2, y2 = two_pos
        x, y = goal_pos
        if x1 < x and x2 < x:
            return True
        if x1 > x and x2 > x:
            return True
        if y1 < y and y2 < y:
            return True
        if y1 > y and y2 > y:
            return True
        return False

    def calc_mht_dist(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)

    def is_on_line_with_goal(self, pos, goal_pos):
        x1, y1 = pos
        x2, y2 = goal_pos
        return x1 == x2 or y1 == y2

    def calculate_min_step(self, clock=True):
        ones = [idx for idx, agent in enumerate(self.agents) if agent.get_type() == 1]
        twos = [idx for idx, agent in enumerate(self.agents) if agent.get_type() == 2]
        ones_pos = [self.agent_positions[i] for i in ones]
        twos_pos = [self.agent_positions[i] for i in twos]

        min_step = 1e9
        for one in ones_pos:
            for two in twos_pos:
                step = self.calculate_min_step_for_two(
                    one, two, self.goal_pos, clock=clock
                )
                if step < min_step:
                    min_step = step
        return min_step

    # Getting a random location in a grid, excluding certain locations
    def get_random_pos(
        self, width: int, height: int, exclude: List[Tuple[int, int]] = []
    ) -> Tuple[int, int]:
        while True:
            position = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            if position not in exclude:
                return position


class Grid(Controller, Trainer, GridUtil, Visual, IVisual):
    def __init__(
        self,
        width: int,
        height: int,
        agents: List["Agent"],
        storage: "Storage",
    ):
        super().__init__(storage=storage)
        self.width = width
        self.height = height
        self.min_step = 0
        self.step_count = 0

        self.agents: List["Agent"] = agents
        self.idx: int = 0

        self.set_interactive_tiles()
        self.reorder_for_min_step()

    def set_interactive_tiles(self):
        # Assign goal to set position
        self.goal_pos = self.get_random_pos(self.width, self.height)

        # Assign agents to random positions
        self.agent_positions = [
            self.get_random_pos(self.width, self.height) for _ in self.agents
        ]

    # ----- Core Functions ----- #
    def step(self, is_testing=False, ep=0, total_ep=1):
        if self.has_ended():
            return
        if self.idx >= len(self.agents):
            self.step_count += 1
            self.idx = 0
            # random.shuffle(self.agent_idx)

        agent = self.agents[self.idx]
        observed_state = self.extract_state(self.idx)
        choose_best = is_testing or not agent.get_type() in self.learners
        action = agent.choose_action(
            observed_state, choose_best=choose_best, episode_ratio=ep / total_ep
        )
        if debug:
            old_pos = self.agent_positions[self.idx]
        reward, next_state, is_terminal = self.move(self.idx, action)
        if debug:
            print(
                f"Agent {self.idx} of type {agent.get_type()} chose action {action} and moved from {old_pos} to {self.agent_positions[self.idx]}"
            )
            print(f"Agent {self.idx} received a reward of {reward}")

        loss, epsilon = agent.update(
            state=observed_state,
            action=action,
            reward=reward,
            next_state=next_state,
            is_terminal=is_terminal,
        )

        self.idx += 1
        return loss, epsilon

    def move(
        self, idx: int, action: Tuple[int, int]
    ):  # List of actions, in the same order as self.agents
        # Update agent to temporary location according to move
        old_x, old_y = self.agent_positions[idx]
        dx, dy = action
        new_x, new_y = old_x + dx, old_y + dy

        reward = 0

        def clamp(i: int, lower: int, upper: int) -> Tuple[int, int]:
            penalty = -10 if i < lower or i > upper else 0
            return penalty, min(max(i, lower), upper)

        # Retreive reward and new location according to Entity.interaction
        penalty, new_x = clamp(new_x, 0, self.width - 1)
        penalty, new_y = clamp(new_y, 0, self.height - 1)
        reward += penalty
        reward -= 1

        new_pos = new_x, new_y
        agent = self.agents[idx]
        if new_pos == self.goal_pos:
            if agent.have_secret:
                reward += 20
                self.goal_reached = True
        else:
            other_indices = [
                other_idx
                for (other_idx, pos) in enumerate(self.agent_positions)
                if other_idx != idx and pos == new_pos
            ]
            other_agents_diff_type = [
                self.agents[o_idx]
                for o_idx in other_indices
                if self.agents[o_idx].get_type() != self.agents[idx].get_type()
            ]
            if len(other_agents_diff_type) > 0:
                if not self.agents[idx].have_secret:
                    reward += 20
                for agents in other_agents_diff_type + [self.agents[idx]]:
                    agents.have_secret_(True)

        self.agent_positions[idx] = new_pos
        if debug:
            print(f"reward: {reward}")
            print(f"has secret: {agent.have_secret}")
        return reward, self.extract_state(idx), self.goal_reached

    # ----- Private Functions ----- #
    def reorder_for_min_step(self):
        idx_positions = list(enumerate(self.agent_positions))
        idx_positions.sort(
            key=lambda idx_pos: -self.calc_mht_dist(idx_pos[1], self.goal_pos)
        )
        self.agents = [self.agents[i] for i, pos in idx_positions]
        self.agent_positions = [self.agent_positions[i] for i, pos in idx_positions]

    # ----- Public Functions ----- #
    def reset(self):
        self.step_count = 0
        self.goal_reached = False
        self.itr = 0
        self.set_interactive_tiles()
        self.min_step = self.calculate_min_step()
        for agent in self.agents:
            agent.reset()
        self.reorder_for_min_step()

    def extract_state(self, idx):
        if debug:
            print(f"all agent pos: {self.agent_positions}")
            print(f"all types: {[agent.get_type() for agent in self.agents]}")

        type = self.agents[idx].get_type()
        x, y = self.agent_positions[idx]
        x2, y2 = self.get_closest_other_agent(x, y, type)
        x3, y3 = self.get_goal_positions()

        state = torch.zeros(state_size, dtype=dtype)

        state[x * side + y] = 1
        state[side**2 + x2 * side + y2] = 1
        state[side**2 * 2 + x3 * side + y3] = 1
        state[side**2 * 3] = 1 if self.agents[idx].have_secret else 0

        return state

    def get_closest_other_agent(self, x, y, type):
        other_type = 2 if type == 1 else 1
        min_dist = 1e9
        min_x, min_y = -1, -1

        for agent, agent_pos in zip(self.agents, self.agent_positions):
            if agent.get_type() == other_type:
                other_x, other_y = agent_pos
                dist = abs(other_x - x) + abs(other_y - y)
                if dist < min_dist:
                    min_x, min_y = other_x, other_y
                    min_dist = dist

        if debug:
            print(
                f"for current agent {x, y}, closest agent of opposite type of {type} is at {min_x, min_y}"
            )
        return min_x, min_y


# ---- Grid

import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING
from multiprocessing import Process, shared_memory, Pipe

from .view import Graph, TestGraph

if TYPE_CHECKING:
    from .storage import Storage


def draw_graphs(storage: "Storage"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    graph = Graph(storage, fig, axs)


def train(grid: "Grid", connection, ep):
    grid.train(ep)
    # TODO: remove hardcode
    q = grid.agents[0].get_q_table()

    shm = shared_memory.SharedMemory(create=True, size=q.nbytes)
    b = np.ndarray(q.shape, dtype=q.dtype, buffer=shm.buf)
    b[:] = q[:]
    connection.send(shm.name)
    shm.close()


def get_process(storage: "Storage", grid: "Grid"):
    conn1, conn2 = Pipe()
    graph_p = Process(
        target=draw_graphs,
        args=[
            storage,
        ],
    )
    # TODO: remove hardcode
    train_p = Process(target=train, args=[grid, conn2, 1000])
    return graph_p, train_p, conn1


def test(grid: "Grid", ep):
    grid.test(ep)


def draw_test_graph(storage: "Storage"):
    fig, axs = plt.subplots()
    graph = TestGraph(storage, fig, axs)


def get_test_process(storage: "Storage", grid: "Grid", ep=1000):
    graph_p = Process(
        target=draw_test_graph,
        args=[storage],
    )
    test_p = Process(target=test, args=[grid, ep])
    return graph_p, test_p


def get_np_from_name(name):
    existing_shm = shared_memory.SharedMemory(name=name)
    # TODO: remove hardcode
    q = np.ndarray((5**5, 4), buffer=existing_shm.buf)
    s = np.copy(q)
    existing_shm.close()
    existing_shm.unlink()
    return s
