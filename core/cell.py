from .agent import Agent


class Empty:
    def __init__(self, pos):
        x, y = pos
        self.x = x
        self.y = y

    def interact(self, other: Agent):
        return -1, (self.x, self.y)

    def __copy__(self):
        return Empty((self.x, self.y))

    def __deepcopy__(self, memo):
        return self.__copy__()


class Goal(Empty):
    def __init__(self, pos):
        x, y = pos
        self.x = x
        self.y = y
        self.reached = False

    def interact(self, other: Agent):
        if other.has_item() and not self.reached:
            self.reached = True
            return 50, (self.x, self.y)
        else:
            return -1, (self.x, self.y)

    def has_reached(self):
        return self.reached

    def __copy__(self):
        copy = Goal((self.x, self.y))
        copy.reached = self.reached
        return copy

    def __deepcopy__(self, memo):
        return self.__copy__()


class Item(Empty):
    def __init__(self, pos):
        self.taken = False
        x, y = pos
        self.x = x
        self.y = y

    def interact(self, other: Agent):
        if not self.taken and not other.has_item():
            self.taken = True
            return 50, (self.x, self.y)

        return -1, (self.x, self.y)

    def get_pos(self):
        return self.x, self.y

    def __copy__(self):
        copy = Item((self.x, self.y))
        copy.taken = self.taken
        return copy

    def __deepcopy__(self, memo):
        return self.__copy__()


class Wall(Empty):
    def __init__(self, pos, dimensions):
        x, y = pos
        self.x = x
        self.y = y

        width, height = dimensions
        self.new_x = min(width - 1, max(0, x))
        self.new_y = min(height - 1, max(0, y))

    def interact(self, other: Agent):
        return -10, (self.new_x, self.new_y)

    def __copy__(self):
        return Wall((self.x, self.y))

    def __deepcopy__(self, memo):
        return self.__copy__()
