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


def _step_param(
    current_value: Any,
    values: list,
    direction: int,
    rng: random.Random,
) -> Any:
    """
    Step a parameter by `direction` (+1 or -1) positions in its value list.
    For numerical params: index step.
    For categorical params: uniform resample 
    """
    if not _is_numerical(values):
        # Categorical: resample uniformly, excluding current value if possible
        choices = [v for v in values if v != current_value] or values
        return rng.choice(choices)
    try:
        idx = values.index(current_value)
    except ValueError:
        return rng.choice(values)
    new_idx = max(0, min(len(values) - 1, idx + direction))
    return values[new_idx]


def _passthrough_step():
    """An sklearn transformer that does nothing (identity)."""
    return FunctionTransformer()


def _mutate_step(
    child: "PipelineIndividual",
    step: str,
    space: Dict[str, Any],
    rng: random.Random,
    p_mutate: float = 1/3,
) -> None:
    """
    Independently consider one pipeline step for mutation (in-place).

    With probability p_mutate:
      - Heads (p=0.5): randomly resample entire step
      - Tails (p=0.5): roll all parameters, each independently:
                         +1 grid step  (p=1/3)
                         -1 grid step  (p=1/3)
                         no change     (p=1/3)
    """
    if rng.random() >= p_mutate:
        return   # this step is not mutated

    if rng.random() < 0.5:
        # Replace entire step
        new_name = rng.choice(list(space.keys()))
        if step == "selector":
            child.selector = new_name
            child.selector_params = _sample_params(space[new_name], rng)
        elif step == "transformer":
            child.transformer = new_name
            child.transformer_params = _sample_params(space[new_name], rng)
        else:
            child.classifier = new_name
            child.classifier_params = _sample_params(space[new_name], rng)
    else:
        # Sweep all parameters of the current step
        if step == "selector":
            current_name = child.selector
            current_params = child.selector_params
        elif step == "transformer":
            current_name = child.transformer
            current_params = child.transformer_params
        else:
            current_name = child.classifier
            current_params = child.classifier_params

        grid = space[current_name]
        for key, values in grid.items():
            if not isinstance(values, list):
                continue
            roll = rng.random()
            if roll < 1/3:
                current_params[key] = _step_param(current_params[key], values, +1, rng)
            elif roll < 2/3:
                current_params[key] = _step_param(current_params[key], values, -1, rng)
            # else roll >= 2/3: no change


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
          
        """
        if rng is None:
            rng = random.Random()

        child = copy.deepcopy(self)
        child.cv_score = None
        child.oof_predictions = None
        child.n_features_in = None
        child.n_features_out = None

        _mutate_step(child, "selector", SELECTOR_SPACE, rng)
        _mutate_step(child, "transformer", TRANSFORMER_SPACE, rng)
        _mutate_step(child, "classifier", CLASSIFIER_SPACE, rng)

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

        if rng.random() < 1/3:
            child_a.selector, child_b.selector = (
                child_b.selector, child_a.selector)
            child_a.selector_params, child_b.selector_params = (
                child_b.selector_params, child_a.selector_params)

        if rng.random() < 1/3:
            child_a.transformer, child_b.transformer = (
                child_b.transformer, child_a.transformer)
            child_a.transformer_params, child_b.transformer_params = (
                child_b.transformer_params, child_a.transformer_params)

        if rng.random() < 1/3:
            child_a.classifier, child_b.classifier = (
                child_b.classifier, child_a.classifier)
            child_a.classifier_params, child_b.classifier_params = (
                child_b.classifier_params, child_a.classifier_params)


        _mutate_step(child_a, "selector", SELECTOR_SPACE, rng)
        _mutate_step(child_a, "transformer", TRANSFORMER_SPACE, rng)
        _mutate_step(child_a, "classifier", CLASSIFIER_SPACE, rng)

        _mutate_step(child_b, "selector", SELECTOR_SPACE, rng)
        _mutate_step(child_b, "transformer", TRANSFORMER_SPACE, rng)
        _mutate_step(child_b, "classifier", CLASSIFIER_SPACE, rng)

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
