from typing import TYPE_CHECKING, List

from .multithread import get_process, get_test_process, get_np_from_name

if TYPE_CHECKING:
    from .model import Model
    from .c_storage import Storage
    from core import Grid, Agent


class Trainer:
    def __init__(self, max_itr):
        self.max_itr = max_itr

    def bind(
        self, model: "Model", storage: "Storage", grid: "Grid", agents: List["Agent"]
    ):
        self.model = model
        self.storage = storage
        self.grid = grid
        self.agents = agents

    def train(self, itr=1):
        self.model.reset()
        for i in range(itr):
            (
                loss,
                reward,
                epsilon,
            ) = self.train_one_game()
            # self.storage.append_loss_epsilon(loss, epsilon)
            print(f"itr: {i}")

    def test(self, itr=1):
        self.model.reset()
        for i in range(self.max_itr):
            self.storage.test_loss[i] = 0
        for _ in range(itr):
            (
                loss,
                reward,
                epsilon,
            ) = self.train_one_game(learn=False)
            self.storage.append_test_loss(loss)

    def train_one_game(self, learn=True):
        self.model.reset()
        max_reward = self.grid.get_max_reward()

        max_step_count = 50 if learn else 50
        step_count = 0
        while not self.grid.get_state().is_terminal() and step_count < max_step_count:
            self.step(learn)
            step_count += 1

        total_reward = sum(map(lambda a: a.get_total_reward(), self.agents))
        loss = max_reward - total_reward
        return loss, total_reward, self.agents[0].epsilon  # TODO: 0

    def step(self, learn=True):
        if self.grid.get_state().is_terminal():
            return

        state = self.grid.get_state()
        actions = [agent.choose_action(state, explore=learn) for agent in self.agents]
        results = self.grid.move(actions)

        for action, (reward, next_state, terminal), agent in zip(
            actions, results, self.agents
        ):
            if learn:
                agent.update_learn(state, action, reward, next_state, terminal)
            else:
                agent.update(next_state, reward)

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
