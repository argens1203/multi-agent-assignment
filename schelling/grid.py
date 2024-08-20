from grid import Grid


class SchellingGrid(Grid):
    def jump_to_empty(self, agent, new_y, new_x):
        assert self.grid[new_y][new_x] is None

        old_y = agent.get_y()
        old_x = agent.get_x()

        agent.set_y(new_y)
        agent.set_x(new_x)

        self.grid[old_y][old_x], self.grid[new_y][new_x] = (
            self.grid[new_y][new_x],
            self.grid[old_y][old_x],
        )
        self.empty.remove((new_y, new_x))
        self.empty.append((old_y, old_x))

    def get_repr(self):
        return list(
            map(
                lambda row: list(
                    map(lambda grid: 0 if grid is None else grid.get_type(), row)
                ),
                self.grid,
            )
        )
