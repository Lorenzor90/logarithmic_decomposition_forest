"""logarithmic_decomposition_forest

A sparse-table-based data structure that answers range queries for any
associative operation in O(1) after O(n log n) precomputation.

Exported names:
    LogarithmicDecompositionForest  -- the main class
    IdentityError                   -- raised when combine fails for empty ranges
"""

from .core import LogarithmicDecompositionForest, IdentityError

__all__ = ["LogarithmicDecompositionForest", "IdentityError"]
