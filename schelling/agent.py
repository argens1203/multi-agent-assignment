import random
from agent import GridAgent


class SchellingAgent(GridAgent):
    def __init__(self, type, y, x, threshold):
        super().__init__(type, y, x)

        self.threshold = threshold
        self.happy = True
        self.similarity_nearby = 0.0

    # ----------- Perceive -----------

    def perceive(self, grid):
        neighbours = self.get_neighbours(grid)
        total = len(neighbours)

        same_type_neighbours = filter(lambda other: self.same_as(other), neighbours)
        same = len(list(same_type_neighbours))

        if total == 0:
            assert same == 0
            return 1.0

        self.similarity_nearby = same / total
        self.happy = self.similarity_nearby > self.threshold

    def get_neighbours(self, grid):
        neighbours = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                y = (self.y + dy) % grid.get_height()
                x = (self.x + dx) % grid.get_width()
                cell = grid.get_cell(y, x)

                if cell is not None:
                    neighbours.append(cell)

        return neighbours

    def same_as(self, other) -> bool:
        return self.type == other.type

    # ----------- React -----------

    def react(self, grid):
        if self.happy:
            return
        pick = random.choice(grid.get_empty_cells())

        # swapping entries
        y, x = self.y, self.x
        new_y, new_x = pick
        grid.jump_to_empty(self, new_y, new_x)

        return

    # ----------- Getters -----------

    def is_happy(self):
        return self.happy

    def get_similarity_nearby(self):
        return self.similarity_nearby
