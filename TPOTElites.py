"""
TPOTElites.py
----------------
AutoML interface wrapping MAP-Elites pipeline search and ensemble selection
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from map_elites import (
    MAPElitesConfig,
    MAPElitesResult,
    EnsembleResult,
    extract_ensemble,
    run_mapelites,
)
from ensemble_selection import ensemble_predict_proba
from pipeline import PipelineIndividual
from search_space import FAMILIES, COMPRESSION_BINS


class TPOTElites(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        generations: int = 50,
        init_size: int = 50,
        population_size: int = 10,
        mutation_rate: float = 0.8,
        cv: int = 5,
        scoring: str = "accuracy",
        ensemble_size: int = 50,
        min_ensemble_score: float = 0.0,
        random_state: Optional[int] = None,
        verbosity: int = 1,
        log_interval: int = 10,
    ):
        self.generations = generations
        self.init_size = init_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.cv = cv
        self.scoring = scoring
        self.ensemble_size = ensemble_size
        self.min_ensemble_score = min_ensemble_score
        self.random_state = random_state
        self.verbosity = verbosity
        self.log_interval = log_interval



    def fit(self, X, y):

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        config = MAPElitesConfig(
            n_generations=self.generations,
            init_population=self.init_size,
            batch_size=self.population_size,
            p_mutate=self.mutation_rate,
            p_crossover=1.0 - self.mutation_rate,
            cv_folds=self.cv,
            scoring=self.scoring,
            ensemble_size=self.ensemble_size,
            random_state=self.random_state if self.random_state is not None else 42,
            verbose=(self.verbosity >= 2),
            log_interval=self.log_interval,
        )

        # MAP-Elites search
        if self.verbosity >= 1:
            print(f"[TPOTElites] Starting MAP-Elites search "
                  f"({self.generations} generations, "
                  f"population={self.population_size}) …")

        self.search_result_: MAPElitesResult = run_mapelites(X, y, config)

        # Ensemble selection
        if self.verbosity >= 1:
            print(f"[TPOTElites] Running Caruana ensemble selection "
                  f"(size={self.ensemble_size}) …")

        self.ensemble_result_: EnsembleResult = extract_ensemble(
            self.search_result_,
            y,
            ensemble_size=self.ensemble_size,
            scoring=self.scoring,
            random_state=self.random_state if self.random_state is not None else 42,
            min_score=self.min_ensemble_score,
        )

        # Retrain ensemble members on full training data
        if self.verbosity >= 1:
            n_unique = self.ensemble_result_.n_unique_members()
            print(f"[TPOTElites] Retraining {n_unique} ensemble members …")

        self.fitted_pipelines_: List[tuple] = []   # (fitted_pipeline, weight)

        for member, weight in zip(
                self.ensemble_result_.members,
                self.ensemble_result_.weights):
            if weight <= 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe = member.build_sklearn_pipeline()
                    pipe.fit(X, y)
                self.fitted_pipelines_.append((pipe, weight))
            except Exception as exc:
                if self.verbosity >= 2:
                    print(f"  [warn] Retraining failed for {member}: {exc}")
                # Skip failed members; re-normalise weights below

        if not self.fitted_pipelines_:
            raise RuntimeError(
                "All ensemble members failed to retrain. "
                "Try increasing population_size or relaxing min_ensemble_score."
            )

        # Re-normalise weights in case any member was dropped
        total_w = sum(w for _, w in self.fitted_pipelines_)
        self.fitted_pipelines_ = [
            (pipe, w / total_w) for pipe, w in self.fitted_pipelines_
        ]

        if self.verbosity >= 1:
            best = self.search_result_.best_individual()
            oof_score = self.ensemble_result_.final_score()
            print(
                f"[TPOTElites] Done. "
                f"Best single CV={best.cv_score:.4f}  "
                f"Ensemble OOF={oof_score:.4f}  "
                f"Members={len(self.fitted_pipelines_)}"
            )

        return self

    

    def predict_proba(self, X) -> np.ndarray:

        check_is_fitted(self)
        X = check_array(X)

        avg = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for pipe, weight in self.fitted_pipelines_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                avg += pipe.predict_proba(X) * weight

        return avg


    def predict(self, X) -> np.ndarray:
        
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]


    # Print archive grid
    def print_archive(self) -> None:
      
        check_is_fitted(self)
        print("\nMAP-Elites Archive (CV scores)")
        self.search_result_.print_grid()

    # Print ensemble members with weights
    def print_ensemble(self) -> None:
       
        check_is_fitted(self)
        result = self.ensemble_result_
        print(f"\nEnsemble ({result.n_unique_members()} members, "
              f"OOF score={result.final_score():.4f})")
        print(f"{'Weight':>8}  {'Family':<16} {'Compression':<12} Pipeline")
        print("-" * 72)
        for member, weight in zip(result.members, result.weights):
            if weight <= 0:
                continue
            fam, comp = member.descriptor()
            print(f"{weight:>8.3f}  {fam:<16} {comp:<12} {member}")
        print()

    # return all archive members
    def get_archive(self) -> List[PipelineIndividual]:
       
        check_is_fitted(self)
        return self.search_result_.archive_as_list()

    # return the best individual pipeline
    @property
    def fitted_pipeline_(self):
        
        check_is_fitted(self)
        best_pipe, _ = max(self.fitted_pipelines_, key=lambda x: x[1])
        return best_pipe

    # Running ensemble score at each greedy step
    @property
    def ensemble_trajectory_(self) -> List[float]:
        check_is_fitted(self)
        return self.ensemble_result_.trajectory

    # Map elites search stats (per gen)
    @property
    def generation_logs_(self) -> List[dict]:
        check_is_fitted(self)
        return self.search_result_.generation_logs


    # sklearn compatibility
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.classifier_tags is not None:
            tags.classifier_tags.multi_class = True
            tags.classifier_tags.multi_output = False
        return tags
