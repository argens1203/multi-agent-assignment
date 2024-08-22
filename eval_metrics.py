class Eval:
    def __init__(self, env) -> None:
        self.env = env

    def calculate_optimum_distance(self):
        x1, y1 = self.env.agent_position
        x2, y2 = self.env.item_position
        x3, y3 = self.env.B_position
        dist_to_obj = x1 - x2 + y1 - y2
        dist_to_goal = x2 - x3 + y2 - y3

        return dist_to_obj + dist_to_goal
