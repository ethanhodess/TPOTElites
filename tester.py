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

    try:

        task_id = [1, 2, 3]
        num_runs = 21

        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=20, stratify=data.target)


        results = []
        for run_num in task_id:
            ####### TPOT Elites ########
            est = TPOTElites(generations=190, init_size=100, population_size=10, ensemble_size=30, random_state=run_num, verbosity=2)
            est.fit(X_train, y_train)
            ensemble_score = accuracy_score(y_test, est.predict(X_test))

            est.print_archive()
            est.print_ensemble()
            
            best = est.search_result_.best_individual()
            pipe = best.build_sklearn_pipeline()
            pipe.fit(X_train, y_train)
            individual_score = accuracy_score(y_test, pipe.predict(X_test))

            results.append({
                "seed": run_num,
                "ensemble_score": ensemble_score,
                "individual_score": individual_score
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
