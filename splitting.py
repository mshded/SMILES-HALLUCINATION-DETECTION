"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits). It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


N_SPLITS = 5


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into repeated stratified train/val/test subsets.

    The public dataset is small and imbalanced, so a single random split is
    unstable. This strategy keeps the original train/validation/test protocol,
    but repeats it with different seeds and returns five folds for averaging.
    """

    idx = np.arange(len(y))
    relative_val = val_size / (1.0 - test_size)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for split_id in range(N_SPLITS):
        seed = random_state + split_id

        idx_train_val, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=seed,
            stratify=y[idx_train_val],
        )

        splits.append((idx_train, idx_val, idx_test))

    return splits