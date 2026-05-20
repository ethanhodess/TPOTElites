"""
ensemble_selection.py
----------
Greedy forward selection
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def greedy_forward_select(
    oof_predictions: List[np.ndarray],
    y_true: np.ndarray,
    ensemble_size: int = 50,
    scoring: str = "accuracy",
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, List[int], List[float]]:
   
    rng = np.random.RandomState(random_state)
    n_candidates = len(oof_predictions)
    n_samples = oof_predictions[0].shape[0]
    n_classes = oof_predictions[0].shape[1]

    weighted_sum = np.zeros((n_samples, n_classes), dtype=np.float64) # current sum of prediction probas
    temp = np.zeros((n_samples, n_classes), dtype=np.float64) # for candidate ensemble

    ensemble_indices: List[int] = []
    cv_trajectory: List[float] = []

    for step in range(ensemble_size):
        scores = np.full(n_candidates, -np.inf, dtype=np.float64)
        s = len(ensemble_indices)

        for j, pred in enumerate(oof_predictions):
            # temp = (weighted_sum + pred) / (s + 1)
            np.add(weighted_sum, pred, out=temp)
            np.multiply(temp, 1.0 / (s + 1), out=temp)
            scores[j] = _score(temp, y_true, scoring)

        # Break ties randomly
        best_val = np.nanmax(scores)
        all_best = np.where(scores == best_val)[0]
        best = int(rng.choice(all_best))

        best_pred = oof_predictions[best]
        ensemble_indices.append(best)
        cv_trajectory.append(best_val)

        # Update weighted sum
        np.add(weighted_sum, best_pred, out=weighted_sum)

        if n_candidates == 1:
            break

    # Convert counts to weights
    counts = Counter(ensemble_indices)
    weights = np.zeros(n_candidates, dtype=np.float64)
    for idx, cnt in counts.items():
        weights[idx] = cnt / len(ensemble_indices)

    return weights, ensemble_indices, cv_trajectory


# Scoring helper
def _score(avg_proba: np.ndarray, y_true: np.ndarray, scoring: str) -> float:
    if scoring == "roc_auc" and avg_proba.shape[1] == 2:
        return roc_auc_score(y_true, avg_proba[:, 1])
    else:
        pred_labels = np.argmax(avg_proba, axis=1)
        return accuracy_score(y_true, pred_labels)


def ensemble_predict_proba(
    predictions: List[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(predictions[0], dtype=np.float64)
    tmp = np.empty_like(predictions[0], dtype=np.float64)
    for pred, w in zip(predictions, weights):
        if w > 0:
            np.multiply(pred, w, out=tmp)
            np.add(out, tmp, out=out)
    return out


def ensemble_predict(
    predictions: List[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    proba = ensemble_predict_proba(predictions, weights)
    return np.argmax(proba, axis=1)
