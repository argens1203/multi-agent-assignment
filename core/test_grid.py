import unittest
from .grid import GridUtil


class DummyAgent:
    def __init__(self, type):
        self.type = type

    def get_type(self):
        return self.type


class TestGridUtil(unittest.TestCase):
    def setUp(self):
        self.util = GridUtil()
        self.util.agents = [DummyAgent(1), DummyAgent(1), DummyAgent(2), DummyAgent(2)]
        self.util.agent_positions = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.util.goal_pos = (4, 4)

    ######################## Test init ########################
    def test_goal_line_can_cut(self):
        test_cases = [
            [(0, 0), (0, 4), (1, 3), True],
            [(0, 3), (1, 4), (2, 2), True],
            [(2, 2), (3, 1), (4, 1), True],
            [(0, 5), (1, 1), (2, 3), True],
            [(1, 3), (5, 4), (3, 5), True],
            [(1, 1), (3, 5), (3, 3), False],
            [(5, 1), (3, 5), (3, 3), False],
            [(0, 0), (4, 4), (2, 2), False],
            [(1, 3), (4, 0), (3, 3), False],
            [(1, 3), (4, 4), (3, 3), False],
            [(1, 3), (5, 3), (3, 3), False],
            [(3, 0), (3, 6), (3, 3), False],
        ]
        util = GridUtil()
        for one, two, goal, expected in test_cases:
            res = util.line_passing_through_goal_can_cut(one, two, goal)
            self.assertEqual(res, expected)

    def test_calc_mht_dist(self):
        test_cases = [
            [(5, 1), (3, 5), 6],
            [(0, 0), (4, 4), 8],
            [(1, 3), (4, 0), 6],
            [(1, 3), (4, 4), 4],
        ]
        util = GridUtil()
        for pos1, pos2, dist in test_cases:
            res = util.calc_mht_dist(pos1, pos2)
            self.assertEqual(res, dist)

    def test_is_on_line_with_goal(self):
        test_cases = [
            [(0, 5), (5, 5), True],
            [(3, 4), (3, 1), True],
            [(0, 0), (2, 1), False],
            [(3, 4), (0, 1), False],
        ]
        util = GridUtil()
        for pos, goal, expected in test_cases:
            res = util.is_on_line_with_goal(pos, goal)
            self.assertEqual(res, expected)

    def test_calculate_min_step_for_two_simple(self):
        test_cases = [
            [(0, 3), (5, 3), (3, 3), 4],  # Diff side same line
            [(1, 2), (3, 5), (1, 4), 4],  # Diff side one on line
            [(0, 0), (3, 3), (0, 3), 5],  # Perpendicular line
            [(1, 1), (3, 3), (2, 2), 4],  # Non-touching quadrant
            [(2, 3), (5, 2), (0, 4), 7],  # Touching quadrant
        ]
        util = GridUtil()
        for pos1, pos2, goal_pos, expected in test_cases:
            res = util.calculate_min_step_for_two(pos1, pos2, goal_pos)
            self.assertEqual(res, expected)

    def test_calculate_min_step_for_two_clock(self):
        test_cases = [
            [(0, 3), (5, 3), (3, 3), 3],  # Diff side same line
            [(1, 2), (3, 5), (1, 4), 3],  # Diff side one on line
            [(0, 0), (3, 3), (0, 3), 4],  # Perpendicular line
            [(0, 2), (2, 1), (0, 1), 2],  # Perpendicular line, distance differ by 1
            [(0, 2), (3, 1), (0, 1), 3],  # Perpendicular line, distance differ by 2
            [(1, 1), (3, 3), (2, 2), 3],  # Non-touching quadrant
            [(2, 3), (5, 2), (0, 4), 6],  # Touching quadrant
            [(0, 1), (2, 3), (0, 4), 3],  # Touching quadrant same distance
            [(0, 1), (3, 3), (0, 4), 3],  # Touch quadrant, distance dffer by 1
            [(0, 1), (0, 2), (0, 0), 1],  # Simple clock case
        ]
        util = GridUtil()
        for pos1, pos2, goal_pos, expected in test_cases:
            res = util.calculate_min_step_for_two(pos1, pos2, goal_pos, clock=True)
            self.assertEqual(res, expected)
