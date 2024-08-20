from agent import Agent


class A1Agent(Agent):
    def __init__(self):
        self.grid = None

    def set_grid(self, grid):
        self.grid = grid

    def perceive(self):
        pass

    def react(self):
        pass
