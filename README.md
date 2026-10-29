# logarithmic_decomposition_forest

A sparse-table data structure that answers range queries for any associative binary operation in O(1) after O(n log n) precomputation.

```python
from logarithmic_decomposition_forest import LogarithmicDecompositionForest

forest = LogarithmicDecompositionForest([3, 1, 4, 1, 5, 9, 2, 6], min)
forest.query(2, 7)  # -> 1, the minimum of [4, 1, 5, 9, 2]
```

The constructor takes the source sequence, an associative `combine(a, b)` callable, and an optional `identity` element. `query(left, right)` folds the half-open range `[left, right)`. For a non-empty range it performs exactly two `combine` calls; for an empty range (`left == right`) it returns the identity, and raises `IdentityError` if none was supplied.

## Why this exists

Some range-query problems -- RMQ, range-GCD, range-sum with no updates -- come up repeatedly and the sparse table is the textbook O(1)-query answer. The trade-off is memory: the table stores an answer for every power-of-two block starting at every index, so it uses O(n log n) space. This library keeps the implementation small and dependency-free so it can be dropped into an environment with no network access.

The operation must be **associative** but need not be commutative and need not be idempotent. The query combines a prefix cell and a suffix cell that overlap, which is correct under associativity alone. This means sum and string-concatenation work as well as min and max.

## Edge you will hit

Empty ranges (`query(i, i)`) require an identity element because there is nothing to fold. Pass `identity=` to the constructor if you need them; otherwise an `IdentityError` is raised. Bounds are checked: `left < 0`, `right > len(data)`, or `left > right` raise `IndexError`.
