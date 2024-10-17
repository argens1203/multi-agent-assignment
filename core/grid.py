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


class Cell:
    def __init__(self, pos):
        x, y = pos
        self.x = x
        self.y = y

    # returns: score delta, (new_coordinate_x, new_coordinate_y)
    def interact(self, agent: "Agent") -> Tuple[int, Tuple[int, int]]:
        return -1, (self.x, self.y), False


class Goal(Cell):
    def __init__(self, pos):
        super().__init__(pos)
        self.reached = False

    def interact(self, agent: "Agent"):
        if agent.has_secret() and not self.reached:
            self.reached = True
            return 51, (self.x, self.y), True
        else:
            return -1, (self.x, self.y), False

    def has_reached(self):
        return self.reached


class Item(Cell):
    def __init__(self, pos):
        super().__init__(pos)
        self.taken = False

    def interact(self, agent: "Agent"):
        if not self.taken and not agent.has_item():
            agent.set_has_item(True)
            self.taken = True
            return 51, (self.x, self.y), False

        return -1, (self.x, self.y), False

    def get_pos(self):
        return self.x, self.y


class Wall(Cell):
    def __init__(self, pos, dimensions):
        super().__init__(pos)

        width, height = dimensions
        x, y = pos

        self.new_x = min(width - 1, max(0, x))
        self.new_y = min(height - 1, max(0, y))

    def interact(self, agent: "Agent"):
        return -10, (self.new_x, self.new_y), False


class GridFactory:
    # Getting a random location in a grid, excluding certain locations
    def get_random_pos(
        width: int, height: int, exclude: List[Tuple[int, int]] = []
    ) -> Tuple[int, int]:
        while True:
            position = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            if position not in exclude:
                return position


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
        self.step(testing=True)
        return


class Trainer:
    def __init__(self, storage, **kwargs):
        super().__init__(**kwargs)
        self.storage = storage

    def train(self, itr=1):
        start = datetime.datetime.now()
        print(f"Start Time: {start}")
        self.reset()
        for i in range(itr):
            (loss, reward, epsilon, ml_losses) = self.train_one_game(ep=i, total_ep=itr)
            # self.storage.append_loss_epsilon(loss, epsilon)

            self.storage.append_ml_losses(ml_losses)
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch: {i+1}/{itr} -- Time Elapsed: {datetime.datetime.now() - start}"
                )
        return self.storage.ml_losses

    def test(self, itr=1):
        self.reset()
        self.storage.reset_test_loss()
        # for i in range(self.max_itr):
        #     self.storage.test_loss[i] = 0
        for _ in range(itr):
            (loss, reward, epsilon, _) = self.train_one_game(testing=True)
            # self.storage.append_test_loss(loss)
            self.storage.append_test_loss(loss)
        return self.storage.test_loss

    def train_one_game(self, **kwargs):
        self.reset()
        max_reward = self.get_max_reward()

        max_step_count = 50
        step_count = 0
        ml_losses = []
        while not self.has_ended() and step_count < max_step_count:
            ml_loss = self.step(**kwargs)
            if ml_loss is not None:
                ml_losses.append(ml_loss)
            step_count += 1

        total_reward = sum(map(lambda a: a.get_total_reward(), self.agents))
        loss = max_reward - total_reward
        return loss, total_reward, self.agents[0].epsilon, ml_losses  # TODO: 0

    def test_in_background(self, ep=1000):
        gp, tp = get_test_process(self.storage, self, ep)
        gp.start()
        tp.start()
        gp.join()
        tp.join()

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


class Grid(Controller, Trainer, IVisual):
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
        self.max_reward = 0

        self.env: Dict[Tuple[int, int], Cell] = (
            {}
        )  # TODO: multiple entities in one cell
        self.interactables: Dict[Tuple[int, int], set[Cell]] = {}
        self.lookup: set[Cell] = set()  # Interactive tiles
        self.agents: List["Agent"] = agents
        self.agent_idx: List[int] = list(range(len(agents)))
        self.agent_pointer = 0
        self.agent_positions: List[Tuple[int, int]] = []
        self.init_environment()

    # ----- Init Functions ----- #
    def init_environment(self):
        for x in range(-1, self.width + 1):
            for y in range(-1, self.height + 1):
                if x < 0 or x >= self.width:
                    self.env[(x, y)] = Wall((x, y), (self.width, self.height))
                elif y < 0 or y >= self.height:
                    self.env[(x, y)] = Wall((x, y), (self.width, self.height))
                else:
                    self.env[(x, y)] = Cell((x, y))

    def set_interactive_tiles(self):
        self.lookup.clear()
        used_pos = []

        for x in range(self.width):
            for y in range(self.height):
                self.interactables[(x, y)] = set()
        # TODO: extract repeated code

        # Assign goal to set position
        goal_pos = GridFactory.get_random_pos(self.width, self.height, used_pos)
        # goal_pos = (self.width - 1, self.height - 1)
        goal = Goal(goal_pos)
        # self.env[goal_pos] = goal
        self.lookup.add(goal)
        self.goal = goal
        used_pos.append(goal_pos)
        self.interactables[goal_pos].add(goal)

        # Assign items to a random position in the remaining tiles
        # item_pos = GridFactory.get_random_pos(self.width, self.height, used_pos)
        # item = Item(item_pos)
        # # self.env[item_pos] = item
        # self.lookup.add(item)
        # used_pos.append(item_pos)
        # self.item = item
        # self.interactables[item_pos].add(item)

        # Assign agents to random positions
        self.agent_positions = []
        for agent in self.agents:
            agent_pos = GridFactory.get_random_pos(self.width, self.height, used_pos)
            used_pos.append(agent_pos)
            self.agent_positions.append(agent_pos)
            self.interactables[agent_pos].add(agent)

        self.max_reward = GridUtil.calculate_max_reward(self)

    # ----- Core Functions ----- #
    def step(self, testing=False, ep=0, total_ep=1):
        if self.has_ended():
            return
        if self.agent_pointer >= len(self.agent_idx):
            self.agent_pointer = 0
            # random.shuffle(self.agent_idx)

        idx = self.agent_idx[self.agent_pointer]
        agent = self.agents[idx]

        loss = 0

        state = self.extract_state(idx)

        # Off the job training
        learn = not testing
        if not testing:
            percent = ep / total_ep
            # print(agent.get_type(), percent)
            if percent >= 0.99:
                learn = True
            elif (int(percent * 100) % 2 == 0) and agent.get_type() == 1:
                # print(273)
                learn = True
            elif (int(percent * 100) % 2 == 1) and agent.get_type() == 2:
                # print(279)
                learn = True
            else:
                # print("Hello")
                learn = False

        action = agent.choose_action(
            state, testing=testing, learn=learn, ep=ep, total_ep=total_ep
        )
        if debug:
            print(self.agent_positions[idx])
            print(f"agent {idx} of type {agent.get_type()} is making a move: {action}")
        reward, next_state, terminal = self.move(idx, action)

        # print(f"next state is {next_state[:16], next_state[16:32], next_state[32:]}")
        if learn:
            loss += agent.update_learn(
                state,
                action,
                reward,
                next_state,
                terminal,
            )
        else:
            agent.update(reward)
        self.agent_pointer += 1
        return loss if learn else None

    def interact(self, temp_position: Tuple[int, int], agent: "Agent"):
        reward, new_pos, is_terminal = self.env[temp_position].interact(agent)
        if is_terminal:
            return reward, new_pos, is_terminal

        for interactable in self.interactables[new_pos]:
            if interactable is not agent:
                r, _, it = interactable.interact(agent)
                if it is None:
                    it = False
                if _ is None:
                    pass
                reward += r
                is_terminal = is_terminal or it

        return reward, new_pos, is_terminal

    def move(
        self, idx: int, action: Tuple[int, int]
    ):  # List of actions, in the same order as self.agents
        # Update agent to temporary location according to move
        old_pos = self.agent_positions[idx]
        temp_positions = self.process_action(action, old_pos)

        # Retreive reward and new location according to Entity.interaction
        if debug:
            print(f"temp(before bounce back from walls):{temp_positions}")
        reward_new_positions = self.interact(temp_positions, self.agents[idx])
        # self.env[temp_positions].interact(self.agents[idx])
        rewards, new_positions, is_terminal = reward_new_positions
        if debug:
            print(f"new pos: {new_positions}")
        # Update new positions
        self.agent_positions = [pos for pos in self.agent_positions]
        self.agent_positions[idx] = new_positions
        self.interactables[old_pos].remove(self.agents[idx])
        self.interactables[new_positions].add(self.agents[idx])

        # Return move results, in the same order as self.agents
        if debug:
            print(self.agent_positions[idx])
        return rewards, self.extract_state(idx), is_terminal

    # ----- Private Functions ----- #
    def process_action(
        self, action: Tuple[int, int], agent_position: List[Tuple[int, int]]
    ):
        # Move according to action
        x, y = agent_position
        dx, dy = action
        return x + dx, y + dy

    # ----- Public Functions ----- #
    def reset(self):
        self.init_environment()
        self.set_interactive_tiles()
        for agent in self.agents:
            agent.reset()

    def get_size(self):
        return self.width, self.height

    def get_max_reward(self):
        return self.max_reward

    def get_items(self):
        return [x for x in self.lookup if isinstance(x, Item)]

    def get_untaken_items(self):
        items = self.get_items()
        untaken_items = filter(lambda i: not i.taken, items)
        return [i.get_pos() for i in untaken_items]

    def item_taken(self):
        item = next((x for x in self.lookup if isinstance(x, Item)), [None])
        return item.taken

    def get_item_positions(self):
        return []

    def get_goal_positions(self):
        goal = self.goal
        return goal.x, goal.y

    def get_total_reward(self):
        return sum(map(lambda a: a.get_total_reward(), self.agents))

    def get_agent_info(self) -> List[Tuple[Tuple[int, int], int, bool]]:
        """
        Output: List of
                - Tuple of:
                    - coordinate: (int, int)
                    - type: int (1 or 2)
                    - has_secret: bool
        """
        agent_types = map(lambda agent: agent.get_type(), self.agents)
        has_secrets = map(lambda agent: agent.has_secret(), self.agents)
        return list(zip(self.get_agent_positions(), agent_types, has_secrets))

    def has_ended(self) -> bool:
        return self.goal.has_reached()

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

    def extract_state(self, idx):
        if debug:
            print(f"all agent pos: {self.agent_positions}")
            print(f"all types: {[agent.get_type() for agent in self.agents]}")
        type = self.agents[idx].get_type()
        x, y = self.agent_positions[idx]
        x2, y2 = self.get_closest_other_agent(x, y, type)
        x3, y3 = self.get_goal_positions()
        # print(x, y)
        # print(x2, y2)
        # print(x3, y3)
        # TODO: remove hardcoded item_pos indices
        # return agent_pos, item_pos[0], self.has_item()
        state = torch.zeros(state_size, dtype=dtype)
        state[x * side + y] = 1
        # if not self.item_taken():
        state[side**2 + x2 * side + y2] = 1
        state[side**2 * 2 + x3 * side + y3] = 1
        state[side**2 * 3] = 1 if self.agents[idx].has_secret() else 0

        # print(self.agents[idx].has_secret())
        # print(state)
        # input()
        return state

    def get_agent_positions(self):
        return self.agent_positions


class GridUtil:
    def calculate_max_reward(grid: Grid):
        return 100
        # TODO: can only work with one agent and one item ATM
        x1, y1 = grid.get_agent_positions()[0]
        x2, y2 = grid.get_item_positions()[0]
        x3, y3 = grid.get_goal_positions()

        # Manhanttan distance from agent to obj and obj to goal
        dist_to_obj = abs(x1 - x2) + abs(y1 - y2)
        dist_to_goal = abs(x2 - x3) + abs(y2 - y3)

        # +100 for reward and +2 for 2 unneeded mark deduction when stepping on item and goal respectively
        return (dist_to_obj + dist_to_goal) * -1 + 102


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
