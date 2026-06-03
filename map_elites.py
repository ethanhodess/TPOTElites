"""
map_elites.py
-------------------
MAP-Elites search with ensemble selection after
"""

from __future__ import annotations

import random
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray

from ensemble_selection import greedy_forward_select, ensemble_predict_proba
from evaluation import EvalCache, evaluate
from search_space import COMPRESSION_BINS, FAMILIES
from pipeline import PipelineIndividual



# evaluate one pipeline
@ray.remote(num_cpus=1)
def evaluate_remote(
    individual: PipelineIndividual,
    X,
    y,
    cv_folds: int,
    scoring: str,
    random_state: int,
):

    local_cache = {}

    evaluate(
        individual,
        X,
        y,
        cv=cv_folds,
        scoring=scoring,
        cache=local_cache,
        random_state=random_state,
    )

    return individual

@dataclass
class MAPElitesConfig:

    n_generations: int = 50
    init_population: int = 50       # random individuals to seed the grid
    batch_size: int = 10            # children produced per generation

    p_mutate: float = 0.8
    p_crossover: float = 0.2

    cv_folds: int = 5
    scoring: str = "accuracy"       # "accuracy" | "roc_auc"
    random_state: int = 42

    n_jobs: int = 1

    ensemble_size: int = 50         # Caruana forward selection slots

    verbose: bool = True
    log_interval: int = 10          # print grid summary every N gens


# Archive helpers

Archive = Dict[Tuple[int, int], PipelineIndividual]

def _grid_shape() -> Tuple[int, int]:
    return len(FAMILIES), len(COMPRESSION_BINS)

# Places pipeline in a cell (if it wins)
def _try_place(archive: Archive, individual: PipelineIndividual) -> bool:
    cell = individual.descriptor_index()
    current = archive.get(cell)
    if current is None or individual.cv_score > current.cv_score:
        archive[cell] = individual
        return True
    return False

# randomly pick from elites (usually 2 for crossover)
def _sample_elites(
    archive: Archive,
    n: int,
    rng: random.Random,
) -> List[PipelineIndividual]:
    elites = list(archive.values())
    if not elites:
        return []
    return rng.sample(elites, k=n)



# Map elites search
@dataclass
class MAPElitesResult:
    archive: Archive
    eval_cache: EvalCache
    n_evaluations: int
    wall_time: float
    generation_logs: List[dict] = field(default_factory=list)

    def filled_cells(self) -> int:
        return len(self.archive)

    def grid_shape(self) -> Tuple[int, int]:
        return _grid_shape()

    def best_individual(self) -> Optional[PipelineIndividual]:
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda x: x.cv_score or -np.inf)

    # All archive members sorted by score descending
    def archive_as_list(self) -> List[PipelineIndividual]:
        return sorted(
            self.archive.values(),
            key=lambda x: x.cv_score or -np.inf,
            reverse=True,
        )

    # Print archive grid
    def print_grid(self) -> None:
        n_rows, n_cols = _grid_shape()
        col_w = 10
        header = " " * 14 + "".join(
            f"{b:^{col_w}}" for b in COMPRESSION_BINS)
        print(header)
        print(" " * 14 + "-" * (col_w * n_cols))
        for r, fam in enumerate(FAMILIES):
            row_parts = []
            for c in range(n_cols):
                ind = self.archive.get((r, c))
                if ind is None:
                    row_parts.append(f"{'—':^{col_w}}")
                else:
                    row_parts.append(f"{ind.cv_score:^{col_w}.4f}")
            print(f"{fam:>14}|" + "|".join(row_parts) + "|")
        print()


def run_mapelites(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[MAPElitesConfig] = None,
) -> MAPElitesResult:

    if config is None:
        config = MAPElitesConfig()
    
    if not ray.is_initialized():
        ray.init(
            num_cpus=os.cpu_count(),
            ignore_reinit_error=True,
        )

    rng = random.Random(config.random_state)
    np.random.seed(config.random_state)

    archive: Archive = {}
    cache: EvalCache = {}
    n_evals = 0
    logs = []
    t0 = time.time()

    X_ref = ray.put(X)
    y_ref = ray.put(y)

    # Random initial population
    if config.verbose:
        print(f"TPOT-Elites: initializing with {config.init_population} "
              f"random individuals …")

    init_individuals = [
        PipelineIndividual.random(rng) for _ in range(config.init_population)
    ]

    futures = [
        evaluate_remote.remote(
            ind,
            X_ref,
            y_ref,
            config.cv_folds,
            config.scoring,
            config.random_state,
        )
        for ind in init_individuals
    ]

    evaluated_inds = ray.get(futures)

    for ind in evaluated_inds:
        n_evals += 1
        if ind.cv_score > -np.inf:
            _try_place(archive, ind)

   

    if config.verbose:
        print(f"  Init done. Filled {len(archive)}/{_grid_shape()[0]*_grid_shape()[1]} "
              f"cells. Best: {max((v.cv_score for v in archive.values()), default=0):.4f}")


    # EA loop
    for gen in range(config.n_generations):
        if not archive:
            # No filled cells yet
            children = [PipelineIndividual.random(rng)
                        for _ in range(config.batch_size)]
        else:
            children = _produce_children(archive, config, rng)

        futures = [
            evaluate_remote.remote(
                child,
                X_ref,
                y_ref,
                config.cv_folds,
                config.scoring,
                config.random_state,
            )
            for child in children
        ]
        evaluated_children = ray.get(futures)

        n_placed = 0
        for child in evaluated_children:
            n_evals += 1
            if child.cv_score > -np.inf:
                placed = _try_place(archive, child)
                n_placed += int(placed)

        # Logging
        if archive:
            best_score = max(v.cv_score for v in archive.values())
            mean_score = np.mean([v.cv_score for v in archive.values()])
        else:
            best_score = mean_score = 0.0

        log_entry = {
            "generation": gen,
            "filled_cells": len(archive),
            "n_evaluations": n_evals,
            "best_score": best_score,
            "mean_archive_score": mean_score,
            "placements": n_placed,
        }
        logs.append(log_entry)

        if config.verbose and (gen % config.log_interval == 0
                               or gen == config.n_generations - 1):
            print(f"  Gen {gen:>4} | cells={len(archive):>2}/{_grid_shape()[0]*_grid_shape()[1]} "
                  f"| best={best_score:.4f} | mean={mean_score:.4f} "
                  f"| evals={n_evals} | placed={n_placed}")

    wall_time = time.time() - t0
    if config.verbose:
        print(f"\nSearch complete: {n_evals} evaluations in {wall_time:.1f}s")
        print(f"Archive: {len(archive)} / {_grid_shape()[0]*_grid_shape()[1]} cells filled\n")

    return MAPElitesResult(
        archive=archive,
        eval_cache=cache,
        n_evaluations=n_evals,
        wall_time=wall_time,
        generation_logs=logs,
    )


def _produce_children(
    archive: Archive,
    config: MAPElitesConfig,
    rng: random.Random,
) -> List[PipelineIndividual]:

    children = []
    while len(children) < config.batch_size:
        op = rng.random()
        if op < config.p_mutate or len(archive) < 2:
            # Mutation
            parent = rng.choice(list(archive.values()))
            children.append(parent.mutate(rng))
        else:
            # Crossover
            elites = _sample_elites(archive, 2, rng)
            if len(elites) >= 2 and elites[0] != elites[1]:
                c1, c2 = PipelineIndividual.crossover(elites[0], elites[1], rng)
                children.extend([c1, c2])
            else:
                children.append(elites[0].mutate(rng))
    return children[:config.batch_size]


# Ensemble selection
@dataclass
class EnsembleResult:
    weights: np.ndarray
    selected_indices: List[int]
    trajectory: List[float]
    members: List[PipelineIndividual]   # archive members, parallel to weights
    oof_predictions: List[np.ndarray]   # parallel to members

    def n_unique_members(self) -> int:
        return int(np.sum(self.weights > 0))

    def final_score(self) -> float:
        return self.trajectory[-1] if self.trajectory else 0.0

    def predict_proba(self, X_val_preds: List[np.ndarray]) -> np.ndarray:
        return ensemble_predict_proba(X_val_preds, self.weights)

# Takes GP result and runs greedy forward selection to form ensemble
def extract_ensemble(
    result: MAPElitesResult,
    y: np.ndarray,
    ensemble_size: int = 50,
    scoring: str = "accuracy",
    random_state: int = 42,
    min_score: float = 0.0,
) -> EnsembleResult:

    members = [
        ind for ind in result.archive_as_list()
        if ind.cv_score is not None and ind.cv_score >= min_score
    ]

    if not members:
        raise ValueError("No valid archive members to build ensemble from.")

    oof_preds = [ind.oof_predictions for ind in members]

    weights, indices, trajectory = greedy_forward_select(
        oof_predictions=oof_preds,
        y_true=y,
        ensemble_size=ensemble_size,
        scoring=scoring,
        random_state=random_state,
    )

    return EnsembleResult(
        weights=weights,
        selected_indices=indices,
        trajectory=trajectory,
        members=members,
        oof_predictions=oof_preds,
    )

def extract_ensemble_equal_weight(result: MAPElitesResult) -> EnsembleResult:
    members = result.archive_as_list()
    n = len(members)
    weights = np.ones(n) / n
    oof_preds = [ind.oof_predictions for ind in members]
    
    return EnsembleResult(
        weights=weights,
        selected_indices=list(range(n)),
        trajectory=[],
        members=members,
        oof_predictions=oof_preds,
    )