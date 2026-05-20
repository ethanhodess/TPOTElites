"""
search_space.py
----------
Defines the constrained pipeline search space for selectors, transformers, and classifiers

Defines family mappings for each 
"""

import numpy as np

# Selectors
SELECTOR_SPACE = {
    "Passthrough": {},

    "sklearn.feature_selection.SelectPercentile": {
        "percentile": list(range(10, 100, 10)),
    },

    "sklearn.feature_selection.SelectFwe": {
        "alpha": [0.001, 0.01, 0.05],
    },

    "sklearn.feature_selection.VarianceThreshold": {
        "threshold": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],
    },
}

# Transformers
TRANSFORMER_SPACE = {
    "Passthrough": {},

    "sklearn.preprocessing.StandardScaler": {},

    "sklearn.preprocessing.MinMaxScaler": {},

    "sklearn.preprocessing.RobustScaler": {},

    "sklearn.decomposition.PCA": {
        "svd_solver": ["randomized"],
        "iterated_power": list(range(1, 6)),
    },

    "sklearn.preprocessing.Normalizer": {
        "norm": ["l1", "l2", "max"],
    },

    "sklearn.preprocessing.PolynomialFeatures": {
        "degree": [2],
        "include_bias": [False],
        "interaction_only": [False, True],
    },
}

# Classifiers
CLASSIFIER_SPACE = {
    "sklearn.naive_bayes.GaussianNB": {},

    "sklearn.naive_bayes.BernoulliNB": {
        "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        "fit_prior": [True, False],
    },

    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(1, 11)),
        "min_samples_split": list(range(2, 21)),
        "min_samples_leaf": list(range(1, 21)),
    },

    "sklearn.ensemble.ExtraTreesClassifier": {
        "n_estimators": [100],
        "criterion": ["gini", "entropy"],
        "max_features": list(np.arange(0.1, 1.01, 0.1).round(2)),
        "min_samples_split": list(range(2, 11)),
        "min_samples_leaf": list(range(1, 11)),
        "bootstrap": [True, False],
    },

    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": [100],
        "criterion": ["gini", "entropy"],
        "max_features": list(np.arange(0.1, 1.01, 0.1).round(2)),
        "min_samples_split": list(range(2, 11)),
        "min_samples_leaf": list(range(1, 11)),
        "bootstrap": [True, False],
    },

    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": [100],
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "max_depth": list(range(1, 6)),
        "min_samples_split": list(range(2, 11)),
        "min_samples_leaf": list(range(1, 11)),
        "subsample": list(np.arange(0.5, 1.01, 0.1).round(2)),
        "max_features": list(np.arange(0.1, 1.01, 0.1).round(2)),
    },

    "sklearn.neighbors.KNeighborsClassifier": {
        "n_neighbors": list(range(1, 21)),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },

    "sklearn.svm.SVC": {
        "C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        "kernel": ["rbf", "poly", "sigmoid"],
        "degree": [2, 3],
        "probability": [True],
        "cache_size": [200],
    },

    "sklearn.linear_model.LogisticRegression": {
        "penalty": ["l2"],
        "C": [1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0, 25.0],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },

    "sklearn.linear_model.SGDClassifier": {
        "loss": ["log_loss", "modified_huber"],
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": [1e-4, 1e-3, 1e-2],
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "max_iter": [1000],
    },
}


# Classifier → family mapping
CLASSIFIER_FAMILY = {
    "sklearn.naive_bayes.GaussianNB":                  "probabilistic",
    "sklearn.naive_bayes.BernoulliNB":                 "probabilistic",
    "sklearn.tree.DecisionTreeClassifier":             "tree",
    "sklearn.ensemble.ExtraTreesClassifier":           "tree_ensemble",
    "sklearn.ensemble.RandomForestClassifier":         "tree_ensemble",
    "sklearn.ensemble.GradientBoostingClassifier":     "tree_ensemble",
    "sklearn.neighbors.KNeighborsClassifier":          "neighbor",
    "sklearn.svm.SVC":                                 "kernel",
    "sklearn.linear_model.LogisticRegression":         "linear",
    "sklearn.linear_model.SGDClassifier":              "linear",
}

# classifier families
FAMILIES = ["linear", "probabilistic", "tree", "tree_ensemble", "kernel", "neighbor"]


COMPRESSION_BINS = ["none", "low", "medium", "high"]

# Compression ratio = n_features_out / n_features_in
def compression_bin(ratio: float) -> str:
    if ratio >= 1.0:
        return "none"
    elif ratio > 0.7:
        return "low"
    elif ratio >= 0.4:
        return "medium"
    else:
        return "high"
