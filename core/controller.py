from multiprocessing import Array


class Controller(object):
    # Iterate by number of games
    def __init__(self, game, max_itr):
        self.game = game
        self.timeout = 0.5
        self.auto_reset = True
        self.itr = 0
        self.max_itr = max_itr

        self.iterations = Array("i", range(max_itr))
        self.losses = Array("i", max_itr)
        self.epsilon = Array("f", max_itr)

        self.test_loss = Array("f", max_itr)

    def get_info(self):
        info = self.game.get_agent_info()
        items = self.game.get_untaken_items()
        tot_reward = self.game.get_total_reward()
        max_reward = self.game.get_max_reward()
        return info, items, tot_reward, max_reward

    def set_timeout(self, timeout):
        self.timeout = timeout

    def toggle_auto_reset(self):
        self.auto_reset = not self.auto_reset
        return self.auto_reset

    def next(self):
        if self.game.has_ended() and self.auto_reset:
            self.game.reset()
        self.game.step(learn=False)
        return self.get_info()

    def train(self, itr=1):
        self.game.reset()
        for _ in range(itr):
            (
                loss,
                reward,
                epsilon,
            ) = self.game.train_one_game()
            if self.itr >= self.max_itr:
                self.itr = 0
            self.losses[self.itr] = loss
            self.epsilon[self.itr] = epsilon
            self.itr += 1

    def test(self, itr=1):
        self.game.reset()
        for i in range(self.max_itr):
            self.test_loss[i] = 0
        for _ in range(itr):
            (
                loss,
                reward,
                epsilon,
            ) = self.game.train_one_game(learn=False)
            if self.itr >= self.max_itr:
                self.itr = 0
            self.test_loss[self.itr] = loss
            self.itr += 1

    def get_metrics(self):
        return self.iterations, self.losses, self.epsilon
