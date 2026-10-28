"""Core implementation of the logarithmic decomposition forest.

A sparse table builds ``len(floors(log2(n)))`` layers of prefix/suffix
answers.  Every query [left, right) is covered by at most two cells of the
same layer whose combined span contains the range, giving O(1) lookup.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")


class IdentityError(ValueError):
    """Raised when a query is attempted before an identity element is set."""


class LogarithmicDecompositionForest:
    """Sparse-table forest keyed by block and block-offset.

    Parameters
    ----------
    data
        The immutable source sequence.  A copy is taken so callers cannot
        mutate the table's foundation.
    combine
        An *associative* binary operator ``combine(a, b) -> T``.  It need not
        be commutative; the query preserves left-to-right order so that
        ``combine(prefix_cell, suffix_cell)`` matches folding the raw range.
    identity
        Optional identity element for ``combine``.  Required only if you need
        to query empty ranges (left == right).  For non-empty queries the
        table never needs it.

    Notes
    -----
    Building takes O(n log n) time and space.  Each query is at most
    ``2 * floor(log2(length))`` ``combine`` calls -- O(log n).
    """

    __slots__ = ("_data", "_combine", "_identity", "_tables", "_log")

    def __init__(
        self,
        data: Sequence[T],
        combine: Callable[[T, T], T],
        identity: Optional[T] = None,
    ) -> None:
        self._data: List[T] = list(data)
        self._combine: Callable[[T, T], T] = combine
        self._identity: Optional[T] = identity
        n = len(self._data)
        # Precomputed floor(log2(k)) for every length we might ask about.
        # Going one past ``n`` lets ``query`` look up without bounds checks.
        self._log: List[int] = _build_log_table(n + 1)
        self._tables: List[List[T]] = self._build_tables()

    # ------------------------------------------------------------------ build

    def _build_tables(self) -> List[List[T]]:
        """Build every sparse-table layer.

        Layer ``k`` holds the answer for every block of length ``2**k``:
        ``tables[k][i] = combine(data[i : i + 2**k])``.  Each cell is derived
        from two cells of the previous layer, so building is O(n log n).
        """
        n = len(self._data)
        if n == 0:
            return []
        tables: List[List[T]] = [list(self._data)]
        k = 1
        step = 2
        while step <= n:
            span = 1 << (k - 1)
            prev = tables[k - 1]
            limit = n - step + 1
            row: List[T] = [None] * limit  # type: ignore[list-item]
            for i in range(limit):
                row[i] = self._combine(prev[i], prev[i + span])
            tables.append(row)
            k += 1
            step <<= 1
        return tables

    # ------------------------------------------------------------------ query

    def query(self, left: int, right: int) -> T:
        """Return ``combine(data[left], ..., data[right - 1])``.

        Parameters
        ----------
        left, right
            Half-open interval ``[left, right)``.  ``0 <= left <= right <= n``
            must hold.

        Raises
        ------
        IndexError
            If ``left`` or ``right`` is out of bounds or ``left > right``.
        IdentityError
            If ``left == right`` and no identity was supplied to the
            constructor.
        """
        n = len(self._data)
        if left < 0 or right > n or left > right:
            raise IndexError(f"range [{left}, {right}) out of bounds for length {n}")
        length = right - left
        if length == 0:
            if self._identity is None:
                raise IdentityError(
                    "querying an empty range requires an identity element"
                )
            return self._identity
        # Greedily cover the range with the largest power-of-two blocks,
        # left to right.  Each block is a single table cell, so the total
        # number of ``combine`` calls is at most ``floor(log2(length))``.
        result: Optional[T] = None
        pos = left
        end = right
        while pos < end:
            remaining = end - pos
            k = self._log[remaining]
            span = 1 << k
            # Shrink the block if it would overrun the end (can happen when
            # remaining is not a power of two and the largest block starting
            # at pos would extend past right).
            while pos + span > end:
                k -= 1
                span >>= 1
            cell = self._tables[k][pos]
            result = cell if result is None else self._combine(result, cell)
            pos += span
        # length >= 1 guaranteed above, so result is never None here.
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------ misc

    def __len__(self) -> int:
        return len(self._data)


# The standalone helper is module-private; it keeps ``__slots__`` clean and
# is testable in isolation.


def _build_log_table(max_len: int) -> List[int]:
    """Return ``log`` where ``log[k] = floor(log2(k))`` for ``0 <= k < max_len``.

    Done iteratively in O(max_len): ``log[k] = log[k >> 1] + 1`` for k >= 2.
    The cell for ``k == 0`` is set to ``0`` because ``query`` never reads it
    (an empty range is handled before the log lookup) and a stray ``1`` there
    would mask off-by-one bugs rather than reveal them.
    """
    log: List[int] = [0] * max_len
    for k in range(2, max_len):
        log[k] = log[k >> 1] + 1
    return log
