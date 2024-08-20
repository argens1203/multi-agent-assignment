from agent import GridAgent


class Grid:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.empty = [(y, x) for y in range(height) for x in range(width)]

    def set_cell(self, agent: GridAgent, y, x):
        assert agent is not None

        self.grid[agent.y][agent.x] = agent
        self.empty.remove((y, x))

    def get_cell(self, y, x):
        return self.grid[y][x]

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_empty_cells(self):
        return self.empty
