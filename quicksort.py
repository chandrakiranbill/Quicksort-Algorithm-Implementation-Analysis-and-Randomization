"""
quicksort.py
============

Deterministic and randomized implementations of the Quicksort algorithm.

Both versions use the Lomuto partition scheme described by Cormen et al.
(2022). They differ only in how the pivot is chosen:

* ``quicksort``            - deterministic: the LAST element of the subarray
                             is always the pivot.
* ``randomized_quicksort`` - randomized: a pivot is chosen uniformly at random
                             from the subarray, swapped into the last position,
                             and then the same partition routine is used.

Design notes
------------
* Sorting is done IN PLACE on a Python list; the functions return the same
  list object for convenience so calls can be chained.
* Recursion is applied only to the SMALLER partition, and a ``while`` loop
  handles the larger partition (manual tail-call elimination). This bounds the
  call-stack depth at O(log n) even when the running time degrades to O(n^2),
  so the deterministic version can be benchmarked on sorted input without
  hitting Python's recursion limit.
* An optional ``SortStats`` object counts element comparisons, swaps, and the
  maximum recursion depth so empirical results can be compared to theory
  (the counters add a small constant overhead, so they are off by default).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, MutableSequence, Optional, TypeVar

T = TypeVar("T")

__all__ = [
    "SortStats",
    "partition",
    "quicksort",
    "randomized_partition",
    "randomized_quicksort",
]


@dataclass
class SortStats:
    """Operation counters collected during a single sort."""

    comparisons: int = 0
    swaps: int = 0
    max_depth: int = 0

    def reset(self) -> None:
        self.comparisons = 0
        self.swaps = 0
        self.max_depth = 0


# --------------------------------------------------------------------------- #
#  Partitioning                                                               #
# --------------------------------------------------------------------------- #
def partition(arr: MutableSequence[T], low: int, high: int,
              stats: Optional[SortStats] = None) -> int:
    """Lomuto partition of ``arr[low..high]`` around the pivot ``arr[high]``.

    After the call, every element in ``arr[low..p-1]`` is ``<=`` the pivot,
    ``arr[p]`` is the pivot, and every element in ``arr[p+1..high]`` is
    ``>`` the pivot. The final pivot index ``p`` is returned.

    Loop invariant (for each iteration with index ``j``):
        arr[low..i]     <= pivot
        arr[i+1..j-1]   >  pivot
        arr[high]       == pivot

    Runs in Theta(high - low) time and O(1) extra space.
    """
    pivot = arr[high]
    i = low - 1  # boundary of the "<= pivot" region
    if stats is None:
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
    else:
        for j in range(low, high):
            stats.comparisons += 1
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                stats.swaps += 1
        stats.swaps += 1
    # Place the pivot between the two regions.
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def randomized_partition(arr: MutableSequence[T], low: int, high: int,
                         stats: Optional[SortStats] = None,
                         rng: random.Random = random) -> int:
    """Choose a pivot uniformly at random from ``arr[low..high]``, move it to
    position ``high``, then run the ordinary Lomuto ``partition``."""
    r = rng.randint(low, high)  # inclusive on both ends
    arr[r], arr[high] = arr[high], arr[r]
    return partition(arr, low, high, stats)


# --------------------------------------------------------------------------- #
#  Shared driver                                                              #
# --------------------------------------------------------------------------- #
def _quicksort_driver(arr: MutableSequence[T], low: int, high: int,
                      part: Callable[..., int], stats: Optional[SortStats],
                      depth: int, **kwargs) -> None:
    """Sort ``arr[low..high]`` in place using the partition function ``part``.

    Recurses on the smaller side and loops on the larger side so the stack
    depth never exceeds about log2(n) frames.
    """
    while low < high:
        if stats is not None and depth > stats.max_depth:
            stats.max_depth = depth
        p = part(arr, low, high, stats, **kwargs)
        # Recurse into the smaller partition, iterate over the larger one.
        if p - low < high - p:
            _quicksort_driver(arr, low, p - 1, part, stats, depth + 1, **kwargs)
            low = p + 1
        else:
            _quicksort_driver(arr, p + 1, high, part, stats, depth + 1, **kwargs)
            high = p - 1


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #
def quicksort(arr: List[T], stats: Optional[SortStats] = None) -> List[T]:
    """Deterministic Quicksort (last element as pivot).

    Parameters
    ----------
    arr : list
        The list to sort. It is modified in place.
    stats : SortStats, optional
        If given, comparison/swap/depth counters are recorded into it.

    Returns
    -------
    list
        The same list object, now sorted in non-decreasing order.

    Complexity
    ----------
    Best / average time: Theta(n log n)
    Worst time:          Theta(n^2)  (e.g. already-sorted or reverse-sorted
                                      input, or all keys equal)
    Extra space:         O(log n) stack frames (smaller-side recursion)
    """
    if len(arr) > 1:
        _quicksort_driver(arr, 0, len(arr) - 1, partition, stats, depth=1)
    return arr


def randomized_quicksort(arr: List[T], stats: Optional[SortStats] = None,
                         seed: Optional[int] = None) -> List[T]:
    """Randomized Quicksort (pivot chosen uniformly at random).

    Parameters
    ----------
    arr : list
        The list to sort. It is modified in place.
    stats : SortStats, optional
        If given, comparison/swap/depth counters are recorded into it.
    seed : int, optional
        Seed for a private random generator, for reproducible runs. When
        omitted, Python's global ``random`` module is used.

    Returns
    -------
    list
        The same list object, now sorted in non-decreasing order.

    Complexity
    ----------
    Expected time (for EVERY input): Theta(n log n), about 1.39 n log2 n
                                     comparisons
    Worst time (vanishingly unlikely): Theta(n^2)
    Extra space:                       O(log n) stack frames
    """
    rng = random.Random(seed) if seed is not None else random
    if len(arr) > 1:
        _quicksort_driver(arr, 0, len(arr) - 1, randomized_partition, stats,
                          depth=1, rng=rng)
    return arr


if __name__ == "__main__":
    demo = [38, 27, 43, 3, 9, 82, 10, 3]
    print("Input:               ", demo)
    print("Deterministic sort:  ", quicksort(demo.copy()))
    print("Randomized sort:     ", randomized_quicksort(demo.copy(), seed=42))
