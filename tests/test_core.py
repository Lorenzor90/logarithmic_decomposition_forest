"""Unit tests for logarithmic_decomposition_forest.core."""

import unittest

from logarithmic_decomposition_forest.core import (
    IdentityError,
    LogarithmicDecompositionForest,
    _build_log_table,
)


class TestLogTable(unittest.TestCase):
    def test_small_values(self):
        log = _build_log_table(9)
        self.assertEqual(log[1], 0)
        self.assertEqual(log[2], 1)
        self.assertEqual(log[3], 1)
        self.assertEqual(log[4], 2)
        self.assertEqual(log[7], 2)
        self.assertEqual(log[8], 3)

    def test_zero_entry_is_zero(self):
        log = _build_log_table(4)
        self.assertEqual(log[0], 0)

    def test_empty(self):
        self.assertEqual(_build_log_table(0), [])


class TestMinQueries(unittest.TestCase):
    def setUp(self):
        self.data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]
        self.forest = LogarithmicDecompositionForest(self.data, min, identity=float("inf"))

    def test_single(self):
        for i, v in enumerate(self.data):
            self.assertEqual(self.forest.query(i, i + 1), v)

    def test_full_range(self):
        self.assertEqual(self.forest.query(0, len(self.data)), 1)

    def test_middle(self):
        self.assertEqual(self.forest.query(4, 9), 2)

    def test_power_of_two(self):
        # [1, 4, 1, 5] -> 1
        self.assertEqual(self.forest.query(1, 5), 1)

    def test_two_elements(self):
        self.assertEqual(self.forest.query(0, 2), 1)
        self.assertEqual(self.forest.query(6, 8), 2)


class TestSumQueries(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.forest = LogarithmicDecompositionForest(self.data, lambda a, b: a + b, 0)

    def test_full(self):
        self.assertEqual(self.forest.query(0, 10), 55)

    def test_partial(self):
        self.assertEqual(self.forest.query(2, 7), 3 + 4 + 5 + 6 + 7)

    def test_empty_with_identity(self):
        self.assertEqual(self.forest.query(5, 5), 0)


class TestStringConcat(unittest.TestCase):
    """Confirms non-commutative but associative operations work."""

    def setUp(self):
        self.data = ["a", "b", "c", "d", "e"]
        self.forest = LogarithmicDecompositionForest(self.data, lambda a, b: a + b, "")

    def test_order_preserved(self):
        self.assertEqual(self.forest.query(0, 5), "abcde")
        self.assertEqual(self.forest.query(1, 4), "bcd")
        self.assertEqual(self.forest.query(3, 5), "de")


class TestEdgeCases(unittest.TestCase):
    def test_empty_data(self):
        forest = LogarithmicDecompositionForest([], min, identity=0)
        self.assertEqual(forest.query(0, 0), 0)
        self.assertEqual(len(forest), 0)

    def test_single_element(self):
        forest = LogarithmicDecompositionForest([42], lambda a, b: a + b, 0)
        self.assertEqual(forest.query(0, 1), 42)
        self.assertEqual(forest.query(0, 0), 0)

    def test_empty_range_without_identity(self):
        forest = LogarithmicDecompositionForest([1, 2, 3], min)
        with self.assertRaises(IdentityError):
            forest.query(1, 1)

    def test_out_of_bounds_left(self):
        forest = LogarithmicDecompositionForest([1, 2, 3], min, 0)
        with self.assertRaises(IndexError):
            forest.query(-1, 2)

    def test_out_of_bounds_right(self):
        forest = LogarithmicDecompositionForest([1, 2, 3], min, 0)
        with self.assertRaises(IndexError):
            forest.query(0, 4)

    def test_left_greater_than_right(self):
        forest = LogarithmicDecompositionForest([1, 2, 3], min, 0)
        with self.assertRaises(IndexError):
            forest.query(2, 1)

    def test_data_is_copied(self):
        src = [5, 3, 7]
        forest = LogarithmicDecompositionForest(src, min, 0)
        src[1] = 100
        # Mutating the source list must not affect the forest.
        self.assertEqual(forest.query(0, 3), 3)

    def test_two_element_range_non_power_of_two_length(self):
        # length 3 -> k=1, span=2, overlap is fine for associativity
        forest = LogarithmicDecompositionForest([1, 2, 3, 4], lambda a, b: a + b, 0)
        self.assertEqual(forest.query(0, 3), 6)


class TestMaxQueries(unittest.TestCase):
    def test_power_of_two_length(self):
        data = [5, 2, 8, 1, 9, 3, 7, 4]
        forest = LogarithmicDecompositionForest(data, max, identity=float("-inf"))
        self.assertEqual(forest.query(0, 8), 9)
        self.assertEqual(forest.query(2, 6), 9)


class TestGcd(unittest.TestCase):
    """Another associative, non-idempotent operation from the stdlib."""

    def setUp(self):
        import math
        self.data = [12, 8, 16, 4, 6]
        self.forest = LogarithmicDecompositionForest(self.data, math.gcd, 0)

    def test_whole_range(self):
        self.assertEqual(self.forest.query(0, 5), 2)

    def test_subrange(self):
        # gcd(12, 8, 16) = 4
        self.assertEqual(self.forest.query(0, 3), 4)


if __name__ == "__main__":
    unittest.main()
