import unittest
import tests.helpers
import numpy as np
import src.circleml.core.math as math


class TestDistances(unittest.TestCase):
    def test_euclidean_distance(self):
        cases = [
            (([0, 0], [0, 0]), 0),
            (([0, 0], [1, 1]), np.sqrt(2)),
            (([0, 0], [3, 4]), 5),
            (([1, 1], [1, 1]), 0),
        ]

        for case in cases:
            with self.subTest(case=case):
                case = ((np.array(case[0][0]), np.array(case[0][1])), case[1])
                self.assertEqual(math.euclidean_distance(*case[0]), case[1])

        with self.assertRaises(TypeError):
            math.euclidean_distance([0, 0], [0, 0, 0])

    def test_manhattan_distance(self):
        cases = [
            (([0, 0], [0, 0]), 0),
            (([0, 0], [1, 1]), 2),
            (([0, 0], [3, 4]), 7),
            (([1, 1], [1, 1]), 0),
        ]

        for case in cases:
            with self.subTest(case=case):
                case = ((np.array(case[0][0]), np.array(case[0][1])), case[1])
                self.assertEqual(math.manhattan_distance(*case[0]), case[1])

        with self.assertRaises(TypeError):
            math.manhattan_distance([0, 0], [0, 0, 0])

    def test_hamming_distance(self):
        cases = [
            (([0, 0], [0, 0]), 0),
            (([0, 0], [1, 1]), 2),
            (([0, 0], [3, 4]), 2),
            (([1, 1], [1, 1]), 0),
        ]

        for case in cases:
            with self.subTest(case=case):
                case = ((np.array(case[0][0]), np.array(case[0][1])), case[1])
                self.assertEqual(math.hamming_distance(*case[0]), case[1])

        with self.assertRaises(TypeError):
            math.hamming_distance([0, 0], [0, 0, 0])