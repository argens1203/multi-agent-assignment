import random


class A1Grid:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.goal = None
        self.item = None

    def add_goal(self, y=-1, x=-1):
        # Default location is bottom right
        y = (y + self.height) % self.height
        x = (x + self.width) % self.width
        self.goal = (y, x)

        return (y, x)

    def add_item(self, y=None, x=None):
        y, x = self.get_unoccupied_random(y, x)
        self.item = (y, x)

        return (y, x)

    def add_agent(self, agent, y=None, x=None):
        y, x = self.get_unoccupied_random(y, x)
        self.grid[y][x] = agent
        return (y, x)

    def get_unoccupied_random(self, y, x):
        if y is None:
            y = random.randint(0, self.height - 1)
        if x is None:
            x = random.randint(0, self.width - 1)
        while (y, x) == self.goal:
            print(f"({y},{x}) is occupied, generating new location")
            y, x = self.generate_random_loc()
        return y, x

    def generate_random_loc(self):
        y = random.randint(0, self.height - 1)
        x = random.randint(0, self.width - 1)
        return y, x

    def get_cell(self, y, x):
        return self.grid[y][x]

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_repr(self):
        repr = list(
            map(
                lambda row: list(
                    # map(lambda grid: 0 if grid is None else grid.get_repr(), row)
                    map(
                        lambda grid: 0 if grid is None else -2, row
                    )  # TODO: a more appropriate way than hardcode
                ),
                self.grid,
            )
        )
        y, x = self.goal
        repr[y][x] = 1
        y, x = self.item
        repr[y][x] = -1
        return repr

    # TODO
    def has_ended(self):
        pass

    # TODO
    def get_final_cost(self):
        return None
