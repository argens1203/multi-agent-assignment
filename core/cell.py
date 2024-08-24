from .agent import Agent

class Empty:
    def __init__(self, pos):
        x, y = pos
        self.x = x
        self.y = y

    def interact(self, other: Agent):
        return -1, (self.x, self.y)

class Goal(Empty):
    def __init__(self, pos):
        x, y = pos
        self.x = x
        self.y = y
        self.reached = False

    def interact(self, other: Agent):
        has_item = other.has_item()
        if has_item:
            self.reached = True
            return 50, (self.x, self.y)
        else:
            return -1, (self.x, self.y)
    
    def has_reached(self):
        # print('self.has_reached', self.reached)
        return self.reached

class Item(Empty):
    def __init__(self, pos):
        self.taken = False
        x, y = pos
        self.x = x
        self.y = y
    
    def interact(self, other: Agent):
        # print('interacting')
        if not self.taken:
            self.taken = True
            # print('taken')
            return 50, (self.x, self.y)
        
        return -1, (self.x, self.y)

    def get_pos(self):
        return self.x, self.y

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
