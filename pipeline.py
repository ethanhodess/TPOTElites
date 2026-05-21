"""
pipeline.py
-------------
Defines a constrained pipeline: Selector → Transformer → Classifier

Defines genetic operators
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from search_space import (
    CLASSIFIER_FAMILY,
    CLASSIFIER_SPACE,
    COMPRESSION_BINS,
    FAMILIES,
    SELECTOR_SPACE,
    TRANSFORMER_SPACE,
    compression_bin,
)


# Helpers

def _import_class(dotted_name: str):
    """Import a class from a dotted module path, e.g.
    'sklearn.linear_model.LogisticRegression'."""
    parts = dotted_name.rsplit(".", 1)
    module = import_module(parts[0])
    return getattr(module, parts[1])


def _sample_params(param_grid: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Uniformly sample one value per key from a discrete param grid."""
    return {k: rng.choice(v) if isinstance(v, (list, range)) else v
            for k, v in param_grid.items()}


def _is_numerical(values: list) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in values)


def _adjust_param(
    current_value: Any,
    grid: Dict[str, Any],
    key: str,
    rng: random.Random,
) -> Any:

    values = grid[key]
    if not isinstance(values, list) or not _is_numerical(values):
        # Categorical -- fall back to uniform resample
        return rng.choice(values)
 
    try:
        idx = values.index(current_value)
    except ValueError:
        # Current value not in grid -- fall back to uniform resample
        return rng.choice(values)
 
    # +-1 with boundary clamping
    direction = rng.choice([-1, 1])
    new_idx = max(0, min(len(values) - 1, idx + direction))
    return values[new_idx]


def _passthrough_step():
    """An sklearn transformer that does nothing (identity)."""
    return FunctionTransformer()


# Pipeline individual class
@dataclass
class PipelineIndividual:
    selector: str
    selector_params: Dict[str, Any]
    transformer: str
    transformer_params: Dict[str, Any]
    classifier: str
    classifier_params: Dict[str, Any]

    # Filled in after evaluation
    cv_score: Optional[float] = field(default=None, repr=False)
    oof_predictions: Optional[np.ndarray] = field(default=None, repr=False)
    n_features_in: Optional[int] = field(default=None, repr=False)
    n_features_out: Optional[int] = field(default=None, repr=False)


    # Samples a random pipeline from the search space
    @classmethod
    def random(cls, rng: Optional[random.Random] = None) -> "PipelineIndividual":
        if rng is None:
            rng = random.Random()
        sel = rng.choice(list(SELECTOR_SPACE.keys()))
        trf = rng.choice(list(TRANSFORMER_SPACE.keys()))
        clf = rng.choice(list(CLASSIFIER_SPACE.keys()))
        return cls(
            selector=sel,
            selector_params=_sample_params(SELECTOR_SPACE[sel], rng),
            transformer=trf,
            transformer_params=_sample_params(TRANSFORMER_SPACE[trf], rng),
            classifier=clf,
            classifier_params=_sample_params(CLASSIFIER_SPACE[clf], rng),
        )

    # builds a constrained pipeline 
    def build_sklearn_pipeline(self) -> Pipeline:
        steps = []

        # Selector
        if self.selector == "Passthrough":
            steps.append(("selector", _passthrough_step()))
        else:
            cls_ = _import_class(self.selector)
            steps.append(("selector", cls_(**self.selector_params)))

        # Transformer
        if self.transformer == "Passthrough":
            steps.append(("transformer", _passthrough_step()))
        else:
            cls_ = _import_class(self.transformer)
            steps.append(("transformer", cls_(**self.transformer_params)))

        # Classifier
        cls_ = _import_class(self.classifier)
        steps.append(("classifier", cls_(**self.classifier_params)))

        return Pipeline(steps)

    # Behavior descriptor returns classifier family and compression bucket
    def descriptor(self) -> Tuple[str, str]:
        family = CLASSIFIER_FAMILY.get(self.classifier, "unknown")

        # Measured compression ratio if possible
        if self.n_features_in is not None and self.n_features_out is not None:
            ratio = self.n_features_out / max(self.n_features_in, 1)
            bin = compression_bin(ratio)
        else:
            bin = self._structural_compression_bin()

        return family, bin

    # Estimate compression bin for initial seeding
    def _structural_compression_bin(self) -> str:
        both_pass = (self.selector == "Passthrough"
                     and self.transformer == "Passthrough")
        if both_pass:
            return "none"

        if self.transformer == "sklearn.decomposition.PCA":
            return "high"

        if self.selector != "Passthrough":
            p = self.selector_params.get("percentile", 50)
            return compression_bin(p / 100.0)

        # transformer only (scaler/normalizer) → no compression
        return "none"

    # Index into 2D MAP elites grid
    def descriptor_index(self) -> Tuple[int, int]:
        family, comp = self.descriptor()
        row = FAMILIES.index(family) if family in FAMILIES else 0
        col = COMPRESSION_BINS.index(comp)
        return row, col



    ### Genetic operators

    def mutate(self, rng: Optional[random.Random] = None) -> "PipelineIndividual":
        """
        Return a mutated copy of this individual.

        Mutation strategy:
          A) Replace entire selector with a random one
          B) Replace entire transformer with a random one
          C) Replace entire classifier with a random one
          D) Resample a single hyperparameter of the selector
          E) Resample a single hyperparameter of the transformer
          F) Resample a single hyperparameter of the classifier
          G) Adjust a numerical hyperparameter of the selector by +-1 step
          H) Adjust a numerical hyperparameter of the transformer by +-1 step
          I) Adjust a numerical hyperparameter of the classifier by +-1 step
        """
        if rng is None:
            rng = random.Random()

        child = copy.deepcopy(self)
        # Clear evaluation results — child must be re-evaluated
        child.cv_score = None
        child.oof_predictions = None
        child.n_features_in = None
        child.n_features_out = None

        strategy = rng.choice(["sel_step", "trf_step", "clf_step",
                               "sel_param", "trf_param", "clf_param",
                               "sel_adjust", "trf_adjust", "clf_adjust"])

        if strategy == "sel_step":
            child.selector = rng.choice(list(SELECTOR_SPACE.keys()))
            child.selector_params = _sample_params(
                SELECTOR_SPACE[child.selector], rng)

        elif strategy == "trf_step":
            child.transformer = rng.choice(list(TRANSFORMER_SPACE.keys()))
            child.transformer_params = _sample_params(
                TRANSFORMER_SPACE[child.transformer], rng)

        elif strategy == "clf_step":
            child.classifier = rng.choice(list(CLASSIFIER_SPACE.keys()))
            child.classifier_params = _sample_params(
                CLASSIFIER_SPACE[child.classifier], rng)

        elif strategy == "sel_param":
            grid = SELECTOR_SPACE[child.selector]
            if grid:
                key = rng.choice(list(grid.keys()))
                child.selector_params[key] = rng.choice(grid[key])

        elif strategy == "trf_param":
            grid = TRANSFORMER_SPACE[child.transformer]
            if grid:
                key = rng.choice(list(grid.keys()))
                child.transformer_params[key] = rng.choice(grid[key])

        elif strategy == "clf_param":
            grid = CLASSIFIER_SPACE[child.classifier]
            if grid:
                key = rng.choice(list(grid.keys()))
                child.classifier_params[key] = rng.choice(grid[key])

        elif strategy == "sel_adjust":
            grid = SELECTOR_SPACE[child.selector]
            if grid:
                key = rng.choice(list(grid.keys()))
                child.selector_params[key] = _adjust_param(
                    child.selector_params[key], grid, key, rng)
 
        elif strategy == "trf_adjust":
            grid = TRANSFORMER_SPACE[child.transformer]
            if grid:
                key = rng.choice(list(grid.keys()))
                child.transformer_params[key] = _adjust_param(
                    child.transformer_params[key], grid, key, rng)
 
        elif strategy == "clf_adjust":
            grid = CLASSIFIER_SPACE[child.classifier]
            if grid:
                key = rng.choice(list(grid.keys()))
                child.classifier_params[key] = _adjust_param(
                    child.classifier_params[key], grid, key, rng)

        return child

    # swaps one step between two parents and returns two children
    @staticmethod
    def crossover(
        parent_a: "PipelineIndividual",
        parent_b: "PipelineIndividual",
        rng: Optional[random.Random] = None,
    ) -> Tuple["PipelineIndividual", "PipelineIndividual"]:
    
        if rng is None:
            rng = random.Random()

        child_a = copy.deepcopy(parent_a)
        child_b = copy.deepcopy(parent_b)

        # reset
        for c in (child_a, child_b):
            c.cv_score = None
            c.oof_predictions = None
            c.n_features_in = None
            c.n_features_out = None

        point = rng.choice(["selector", "transformer", "classifier"])

        if point == "selector":
            child_a.selector, child_b.selector = (
                child_b.selector, child_a.selector)
            child_a.selector_params, child_b.selector_params = (
                child_b.selector_params, child_a.selector_params)

        elif point == "transformer":
            child_a.transformer, child_b.transformer = (
                child_b.transformer, child_a.transformer)
            child_a.transformer_params, child_b.transformer_params = (
                child_b.transformer_params, child_a.transformer_params)

        else:
            child_a.classifier, child_b.classifier = (
                child_b.classifier, child_a.classifier)
            child_a.classifier_params, child_b.classifier_params = (
                child_b.classifier_params, child_a.classifier_params)

        return child_a, child_b



    ### Hashing / equality  (for evaluation cache key)

    def _config_dict(self) -> dict:
        return {
            "selector": self.selector,
            "selector_params": self.selector_params,
            "transformer": self.transformer,
            "transformer_params": self.transformer_params,
            "classifier": self.classifier,
            "classifier_params": self.classifier_params,
        }

    # Stable SHA-1 of the pipeline configuration (ignores eval results)
    def config_hash(self) -> str:
        raw = json.dumps(self._config_dict(), sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def __hash__(self) -> int:
        return hash(self.config_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipelineIndividual):
            return NotImplemented
        return self.config_hash() == other.config_hash()

    def __repr__(self) -> str:
        sel = self.selector.split(".")[-1]
        trf = self.transformer.split(".")[-1]
        clf = self.classifier.split(".")[-1]
        score = f"{self.cv_score:.4f}" if self.cv_score is not None else "?"
        return f"Pipeline({sel} → {trf} → {clf}, score={score})"
