import openml
import tpot
import sklearn
import traceback
import dill as pickle
import os
import random
import numpy as np
from tpot.search_spaces.pipelines import ChoicePipeline, SequentialPipeline
from estimator_node_gradual import EstimatorNodeGradual
import pandas as pd
import argparse
import ray

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, accuracy_score)
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

# defines a constrained search space with only three steps
def get_pipeline_space(seed):
    return tpot.search_spaces.pipelines.SequentialPipeline([
        tpot.config.get_search_space(
            ["selectors_classification", "Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot.config.get_search_space(
            ["transformers", "Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot.config.get_search_space("classifiers", random_state=seed, base_node=EstimatorNodeGradual)])

def get_cv_predictions(estimator, X_train, y_train, cv_splits, random_state):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_preds = np.empty(len(y_train), dtype=int)

    for train_idx, valid_idx in cv.split(X_train, y_train):
        est_clone = clone(estimator) 

        try:
            est_clone.fit(X_train[train_idx], y_train[train_idx])
            cv_preds[valid_idx] = est_clone.predict(X_train[valid_idx])
        except Exception as E:
            print('pipeline failed')

    return cv_preds


def get_cv_probas(estimator, X_train, y_train, cv_splits, random_state):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    n_classes = len(np.unique(y_train))
    cv_probas = np.zeros((len(y_train), n_classes))

    for train_idx, valid_idx in cv.split(X_train, y_train):
        est_clone = clone(estimator)
        
        try:
            est_clone.fit(X_train[train_idx], y_train[train_idx])
            cv_probas[valid_idx] = est_clone.predict_proba(X_train[valid_idx])
        except Exception as E:
            print('pipeline failed')       
    return cv_probas

    
@ray.remote
def _ray_get_preds(i, filtered_eval_inds, X_train, y_train, seed):
    # run full CV and return OOF predictions (for clustering pruning)
    est = filtered_eval_inds.iloc[i, 10]
    oof_preds = get_cv_predictions(
        est, X_train, y_train, cv_splits=3, random_state=seed + 105
    )
    return oof_preds


@ray.remote
def _ray_get_probas(estimator, X_train, y_train, seed):
    # run full CV probas (for ensemble step)
    return get_cv_probas(estimator, X_train, y_train, cv_splits=3, random_state=seed + 105)


def clean_eval_inds(eval_inds):
    # filter out the broken pipelines
    filtered_eval_inds = eval_inds[eval_inds["roc_auc_score"].notna()]
    print("length of filtered eval_inds:", len(filtered_eval_inds))
    return filtered_eval_inds


def greedy_forward_search(filtered_eval_inds, X_train, y_train, seed):
    X_ref = ray.put(X_train)
    y_ref = ray.put(y_train)

    estimators = filtered_eval_inds.iloc[:, 10].tolist()
   
    futures = {
        est: _ray_get_probas.remote(ray.put(est), X_ref, y_ref, seed) for est in estimators
    }
    
    est_cv_probas = {est: ray.get(fut) for est, fut in futures.items()}

    # remove bad estimators (pipeline failed during CV)
    failed = [est for est, probas in est_cv_probas.items() if np.all(probas == 0)]
    if failed:
        print(f"dropping {len(failed)} estimators with failed CV probas")
    est_cv_probas = {est: probas for est, probas in est_cv_probas.items() if not np.all(probas == 0)}
    estimators = list(est_cv_probas.keys())

    temp_ensemble = []

    for i in range(50):
        best_candidate = None
        best_candidate_acc = 0


        for est in estimators:

            # test ensemble CV accuracy when each candidate is added
            candidate_probas = [est_cv_probas[e] for e in temp_ensemble + [est]]
            temp_preds = combine_preds(candidate_probas)
            temp_acc = accuracy_score(y_train, temp_preds)

            if temp_acc > best_candidate_acc:
                best_candidate = est
                best_candidate_acc = temp_acc
        
        print(f"ensemble acc changes to {best_candidate_acc:.4f}")
        temp_ensemble.append(best_candidate)


    print(f"FINAL ensemble size: {len(temp_ensemble)}")

    final_ensemble = [est.fit(X_train, y_train) for est in temp_ensemble]
    return final_ensemble



def vote_soft(estimators, X_test, weights=None):
    probas = np.stack([est.predict_proba(X_test) for est in estimators])
    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)
        probas *= weights
    
    return np.argmax(probas.sum(axis=0), axis=1)

def vote_soft_proba(estimators, X_test, weights=None):
    probas = np.stack([est.predict_proba(X_test) for est in estimators])

    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)
        return np.sum(probas * weights, axis=0) / np.sum(weights)

    return np.mean(probas, axis=0)

def combine_preds(proba_list, weights=None):
    probas = np.stack(proba_list, axis=0)

    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)  # (n_models, 1, 1)
        probas = probas * weights

    # Average (or weighted sum) across models
    avg_proba = probas.sum(axis=0) / (weights.sum() if weights is not None else len(proba_list))

    # Pick the class with max probability
    return np.argmax(avg_proba, axis=1)


def main():
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--n_jobs", default=30,
                        required=False, nargs='?')
    # where to save the results/models
    parser.add_argument("-s", "--savepath",
                        default="results_tables", required=False, nargs='?')
    # number of total runs for each experiment
    parser.add_argument("-r", "--num_runs", default=1,
                        required=False, nargs='?')
    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)

    save_folder = base_save_folder

    ray.init()

    try:

        task_ids = [359975, 146820, 190137, 359958, 359968, 359962, 
                    359955, 190411, 359960, 359974, 2073, 168784]
        
        num_runs = 21

        jobs = [(tid, run) for tid in task_ids for run in range(num_runs)]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_id, run_num = jobs[array_id]

        constrained_search_space = get_pipeline_space(seed=run_num)

        full_results = []

        print("task id:", task_id, "run num:", run_num)

        # load the data
        data = pd.read_csv(f'/common/hodesse/hpc_test/TPOTElites/openml_271/task_{task_id}.csv')
        with open(f'/common/hodesse/hpc_test/TPOTElites/openml_271/task_{task_id}_categorical_indicator.pkl', "rb") as f:
            cat_ind = pickle.load(f)

        data.columns = data.columns.str.strip().str.lower()

        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]   

        if len(cat_ind) == data.shape[1]:
            cat_ind = cat_ind[:-1]

        assert len(cat_ind) == X.shape[1]

        cat_cols = X.columns[cat_ind]
        num_cols = X.columns.difference(cat_cols)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,
            random_state=run_num, stratify=y
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test  = preprocessor.transform(X_test)

        y_train = y_train.to_numpy()
        y_test  = y_test.to_numpy()

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test  = le.transform(y_test)

        
        est = tpot.TPOTEstimator(search_space=constrained_search_space, generations=0, population_size=2000, cv=5, n_jobs=n_jobs, max_time_mins=180,
                                 random_state=run_num, verbose=2, classification=True, scorers=['roc_auc_ovr', tpot.objectives.complexity_scorer], scorers_weights=[1, -1])
        est.fit(X_train, y_train)
        eval_inds = est.evaluated_individuals
            

        filtered_eval_inds = clean_eval_inds(eval_inds)
        top70 = filtered_eval_inds.nlargest(70, "roc_auc_score")


        ensemble_5000 = greedy_forward_search(filtered_eval_inds, X_train, y_train, run_num)
        ensemble_70 = greedy_forward_search(top70, X_train, y_train, run_num)

        ensemble_test_proba_5000 = vote_soft_proba(estimators=ensemble_5000, X_test=X_test)

        if len(np.unique(y_test)) == 2:
            ensemble_test_auroc_5000 = roc_auc_score(
                y_test,
                ensemble_test_proba_5000[:, 1]
            )
        else:
            ensemble_test_auroc_5000 = roc_auc_score(
                y_test,
                ensemble_test_proba_5000,
                multi_class="ovr",
                average="macro"
            )

        ensemble_test_proba_70 = vote_soft_proba(estimators=ensemble_70, X_test=X_test)

        if len(np.unique(y_test)) == 2:
            ensemble_test_auroc_70 = roc_auc_score(
                y_test,
                ensemble_test_proba_70[:, 1]
            )
        else:
            ensemble_test_auroc_70 = roc_auc_score(
                y_test,
                ensemble_test_proba_70,
                multi_class="ovr",
                average="macro"
            )

        full_results.append({"task id": task_id,
                            "run #": run_num,
                            "ensemble_5000": ensemble_test_auroc_5000,
                            "ensemble_70": ensemble_test_auroc_70
                            })

        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_csv(os.path.join(save_folder, (f'random_baselines_{task_id}_#{run_num}.csv')), index=False)

    except Exception as e:
        trace = traceback.format_exc()
        pipeline_failure_dict = {"task_id": task_id,
                                 "run": num_runs, "error": str(e), "trace": trace}
        print("failed on ")
        print(save_folder)
        print(e)
        print(trace)

    ray.shutdown()


if __name__ == '__main__':
    main()
    print('DONE')