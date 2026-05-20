"""
evaluation.py
-------------
Evaluates a pipeline with CV

Measures the feature compression ratio (n_features_out / n_features_in)

Caches results
"""

from __future__ import annotations

import traceback
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

from pipeline import PipelineIndividual


# Evaluation and cache

# (cv_score, oof_predictions, n_features_in, n_features_out)
EvalCache = Dict[str, Tuple[float, np.ndarray, int, int]] 


def evaluate(
    individual: PipelineIndividual,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",       # "accuracy" or "roc_auc"
    cache: Optional[EvalCache] = None,
    random_state: int = 42,
) -> PipelineIndividual:

    key = individual.config_hash()

    if cache is not None and key in cache:
        score, oof, n_in, n_out = cache[key]
        individual.cv_score = score
        individual.oof_predictions = oof
        individual.n_features_in = n_in
        individual.n_features_out = n_out
        return individual

    try:
        pipeline = individual.build_sklearn_pipeline()

        skf = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            oof_preds = cross_val_predict(
                pipeline, X, y,
                cv=skf,
                method="predict_proba",
            )

        # Score
        if scoring == "roc_auc" and len(np.unique(y)) == 2:
            score = roc_auc_score(y, oof_preds[:, 1])
        else:
            score = accuracy_score(y, np.argmax(oof_preds, axis=1))

        n_in, n_out = _measure_compression(pipeline, X, y)

    except Exception:
        # for failed pipelines
        score = -np.inf
        oof_preds = np.zeros((len(y), len(np.unique(y))))
        n_in, n_out = X.shape[1], X.shape[1]

    individual.cv_score = score
    individual.oof_predictions = oof_preds
    individual.n_features_in = n_in
    individual.n_features_out = n_out

    if cache is not None:
        cache[key] = (score, oof_preds, n_in, n_out)

    return individual


def _measure_compression(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[int, int]:

    n_in = X.shape[1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            sub = Pipeline(pipeline.steps[:-1])  # pipeline of just the first two steps
            X_out = sub.fit_transform(X, y)
            n_out = X_out.shape[1]
    except Exception:
        n_out = n_in
    return n_in, n_out


def evaluate_batch(
    individuals: list[PipelineIndividual],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
    cache: Optional[EvalCache] = None,
    random_state: int = 42,
    verbose: bool = False,
) -> list[PipelineIndividual]:

    results = []
    for i, ind in enumerate(individuals):
        ind = evaluate(ind, X, y, cv=cv, scoring=scoring,
                       cache=cache, random_state=random_state)
        results.append(ind)
        if verbose:
            print(f"  [{i+1}/{len(individuals)}] {ind}")
    return results
