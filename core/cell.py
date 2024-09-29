from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


class IInteractable(ABC):
    @abstractmethod
    def interact(self, agent: "Agent") -> Tuple[int, Tuple[int, int]]:
        pass


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
        if agent.has_item() and not self.reached:
            self.reached = True
            return 50, (self.x, self.y), True
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
            return 50, (self.x, self.y), False

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
