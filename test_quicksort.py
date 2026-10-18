"""Unit tests for the deterministic and randomized Quicksort implementations.

Run from the repository root with:
    python -m unittest discover -s tests -v
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quicksort import (  # noqa: E402
    SortStats,
    partition,
    quicksort,
    randomized_quicksort,
)

SORTERS = {
    "deterministic": quicksort,
    "randomized": randomized_quicksort,
}


class TestPartition(unittest.TestCase):
    def test_partition_invariant(self):
        rng = random.Random(1)
        for _ in range(200):
            arr = [rng.randint(-50, 50) for _ in range(rng.randint(1, 40))]
            p = partition(arr, 0, len(arr) - 1)
            pivot = arr[p]
            self.assertTrue(all(x <= pivot for x in arr[:p]))
            self.assertTrue(all(x > pivot for x in arr[p + 1:]))


class TestSorting(unittest.TestCase):
    def check(self, data):
        for name, sorter in SORTERS.items():
            with self.subTest(sorter=name, data=data[:10]):
                arr = list(data)
                result = sorter(arr)
                self.assertEqual(result, sorted(data))
                self.assertIs(result, arr, "sort must be in place")

    def test_edge_cases(self):
        for data in ([], [1], [2, 1], [1, 2], [5, 5, 5, 5], [0, -1, 1]):
            self.check(data)

    def test_sorted_and_reverse(self):
        self.check(list(range(500)))
        self.check(list(range(500, 0, -1)))

    def test_duplicates(self):
        rng = random.Random(7)
        self.check([rng.randint(0, 5) for _ in range(300)])

    def test_random_lists(self):
        rng = random.Random(42)
        for n in (3, 10, 50, 200, 1000):
            for _ in range(5):
                self.check([rng.randint(-10**6, 10**6) for _ in range(n)])

    def test_floats_and_strings(self):
        self.check([3.5, -2.25, 0.0, 1e9, -1e-9])
        self.check(["pear", "apple", "fig", "banana", "apple"])

    def test_stack_depth_is_logarithmic(self):
        # Even on the deterministic worst case (sorted input) the smaller-side
        # recursion keeps the stack depth tiny.
        stats = SortStats()
        quicksort(list(range(3000)), stats)
        self.assertLessEqual(stats.max_depth, 2)

    def test_worst_case_comparisons(self):
        # Sorted input forces exactly n(n-1)/2 comparisons for Lomuto/last-pivot.
        n = 400
        stats = SortStats()
        quicksort(list(range(n)), stats)
        self.assertEqual(stats.comparisons, n * (n - 1) // 2)

    def test_randomized_is_reproducible_with_seed(self):
        s1, s2 = SortStats(), SortStats()
        randomized_quicksort(list(range(1000)), s1, seed=123)
        randomized_quicksort(list(range(1000)), s2, seed=123)
        self.assertEqual(s1.comparisons, s2.comparisons)


if __name__ == "__main__":
    unittest.main()
