from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from TPOTElites import TPOTElites
from sklearn.metrics import (roc_auc_score, accuracy_score)
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import dill as pickle
import argparse
import os
import traceback
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


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


    def compute_auroc(model, X_test, y_test):
        y_proba = model.predict_proba(X_test)
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            return roc_auc_score(y_test, y_proba[:, 1])
        else:
            return roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro"
            )

    try:

        # task_ids = [2073, 7593, 10090, 146818, 146820, 167120, 168350, 
        #     168757, 168784, 168909, 168910, 168911, 189354, 189355, 
        #     189922, 190137, 190146, 190392, 190410, 190411, 190412, 
        #     211979, 211986, 359953, 359955, 359956, 359957, 359958, 
        #     359959, 359960, 359961, 359962, 359963, 359964, 359965, 359966, 
        #     359967, 359968, 359969, 359970, 359971, 359972, 359973, 359974, 
        #     359975, 359976, 359977, 359979, 359980, 359981, 359982, 
        #     359984, 359985, 359987, 359990, 
        #     359992, 359994]

        # task_ids = [2073, 146818, 146820, 168350, 168757, 168784, 168911,
        #             190137, 190146, 190411, 359955, 359956, 359957, 359958,
        #             359959, 359962, 359963, 359964, 359965, 359968, 
        #             359971, 359972, 359974, 359975]

        task_ids = [359975, 146820, 190137, 359958, 359966, 359968, 359962, 
                    359955, 190411, 359960, 359974, 2073, 168784, 359969, 359964, 359970]
        
        num_runs = 21

        jobs = [(tid, run) for tid in task_ids for run in range(num_runs)]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_id, run_num = jobs[array_id]

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


        results = []

        est = TPOTElites(generations=0, init_size=2000, population_size=50, ensemble_size=50, random_state=run_num, verbosity=2)
        est.fit(X_train, y_train)
        ensemble_score = compute_auroc(est, X_test, y_test)
        #ensemble_score_unweighted = accuracy_score(y_test, est.predict_unweighted(X_test))

        est.print_archive()
        est.print_ensemble()

        for i, score in enumerate(est.ensemble_trajectory_):
            print(f"step {i+1:>3}: {score:.4f}")
            
        best = est.search_result_.best_individual()
        pipe = best.build_sklearn_pipeline()
        pipe.fit(X_train, y_train)
        individual_score = compute_auroc(pipe, X_test, y_test)

        results.append({
            "task_id": task_id,
            "seed": run_num,
           # "unweighted_ensemble": ensemble_score_unweighted,
            "weighted_ensemble": ensemble_score,
            "individual": individual_score
        })
            
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(save_folder, (f'elites_{task_id}_#{run_num}.csv')), index=False)

    except Exception as e:
        trace = traceback.format_exc()
        pipeline_failure_dict = {"task_id": task_id,
                                 "run": num_runs, "error": str(e), "trace": trace}
        print("failed on ")
        print(save_folder)
        print(e)
        print(trace)


if __name__ == '__main__':
    main()
    print('DONE')
