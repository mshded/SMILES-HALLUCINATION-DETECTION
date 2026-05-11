"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1). Called from ``solution.py``
via ``evaluate.run_evaluation``. All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    The final probe is intentionally small: StandardScaler followed by a
    regularized MLP with two hidden layers. The network is built lazily in
    ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers."""
        self._net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``."""
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors."""
        torch.manual_seed(42)
        np.random.seed(42)

        X_scaled = self._scaler.fit_transform(X)

        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=3e-4,
            weight_decay=5e-3,
        )

        self.train()
        for _ in range(80):
            optimizer.zero_grad()
            logits = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

        self.eval()

        # Важно для инструкции проекта:
        # solution.py при создании финального predictions.csv вызывает только
        # final_probe.fit(...), без fit_hyperparameters(...).
        # Поэтому здесь ставим fallback threshold по labelled data.
        # В evaluation этот threshold затем переопределяется validation tuning.
        self._tune_threshold_by_accuracy(X, y)

        return self

    def _tune_threshold_by_accuracy(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":
        """Tune the decision threshold by validation accuracy."""
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )

        best_threshold = 0.5
        best_accuracy = accuracy_score(
            y_val,
            (probs >= best_threshold).astype(int),
        )

        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = accuracy_score(y_val, y_pred_t)
            if score > best_accuracy:
                best_accuracy = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy."""
        return self._tune_threshold_by_accuracy(X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()

        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)