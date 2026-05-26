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

    # "sklearn.preprocessing.PolynomialFeatures": {
    #     "degree": [2],
    #     "include_bias": [False],
    #     "interaction_only": [False, True],
    # },
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

    # "poly", "sigmoid" kernels removed for runtime test
    "sklearn.svm.SVC": {
        "C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        "kernel": ["rbf"],
        "degree": [2, 3],
        "probability": [True],
        "cache_size": [500],
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

    "sklearn.neural_network.MLPClassifier": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100), (50, 50, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-2],
        "max_iter": [500],
        "early_stopping": [True],
    },

    "sklearn.discriminant_analysis.LinearDiscriminantAnalysis": {
        "solver": ["lsqr"],
        "shrinkage": [None, "auto", 0.1, 0.5, 0.9],
    },
 
    "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis": {
        "reg_param": [0.1, 0.3, 0.5, 0.7, 0.9],   
    },

    "sklearn.ensemble.AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
}


# Classifier → family mapping
CLASSIFIER_FAMILY = {
    "sklearn.naive_bayes.GaussianNB":                               "GaussianNB",
    "sklearn.naive_bayes.BernoulliNB":                              "BernoulliNB",
    "sklearn.tree.DecisionTreeClassifier":                          "DecisionTree",
    "sklearn.ensemble.ExtraTreesClassifier":                        "ExtraTrees",
    "sklearn.ensemble.RandomForestClassifier":                      "RandomForest",
    "sklearn.ensemble.GradientBoostingClassifier":                  "GB",
    "sklearn.neighbors.KNeighborsClassifier":                       "KNeighbors",
    "sklearn.svm.SVC":                                              "SVC",
    "sklearn.linear_model.LogisticRegression":                      "LR",
    "sklearn.linear_model.SGDClassifier":                           "SGD",
    "sklearn.neural_network.MLPClassifier":                         "MLP",
    "sklearn.discriminant_analysis.LinearDiscriminantAnalysis":     "LDA",
    "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis":  "QDA",
    "sklearn.ensemble.AdaBoostClassifier":                          "AdaBoost"

}

# classifier families
FAMILIES = ["GaussianNB", "BernoulliNB", "DecisionTree", "ExtraTrees", "RandomForest", "GB",
            "KNeighbors", "SVC", "LR", "SGD", "MLP", "LDA", "QDA", "AdaBoost"]


COMPRESSION_BINS = ["none", "low", "medium", "medium_high","high"]

# Compression ratio = n_features_out / n_features_in
def compression_bin(ratio: float) -> str:
    if ratio >= 1.0:
        return "none"
    elif ratio >= 0.8:
        return "low"
    elif ratio >= 0.5:
        return "medium"
    elif ratio >= 0.3:
        return "medium_high"
    else:
        return "high"
