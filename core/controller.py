from multiprocessing import Array


class Controller(object):
    # Iterate by number of games
    def __init__(self, game):
        self.game = game
        self.timeout = 0.5
        self.auto_reset = True
        self.itr = 0

        self.iterations = Array("i", range(50000))
        self.losses = Array("i", 50000)
        self.rewards = Array("i", 50000)

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
        for _ in range(itr):
            (
                loss,
                reward,
                max_reward,
            ) = self.game.train_one_game()
            self.losses[self.itr] = loss
            self.rewards[self.itr] = reward
            self.itr += 1

    def get_metrics(self):
        return self.iterations, self.losses, self.rewards
