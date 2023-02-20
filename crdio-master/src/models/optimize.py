import sys
import os
import pandas as pd

import sklearn
from sklearn.model_selection import cross_val_score
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier  # RandomForest Model

from src.features.prepare_modelling import prepare_modelling
import configparser

sys.path.append(os.getcwd())


def main():
    # get paths from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    processed_paths = config["PROCESSED"]
    
    df = pd.read_csv(processed_paths["ALL_FEATURES_TRAIN_PATH"], sep=";")
    X,Y,_ = prepare_modelling(df)

    def objective(trial):

        max_depth = trial.suggest_int("rf_max_depth", 1, 100, log=True)
        max_samples = trial.suggest_float("rf_max_samples", 0, 1)
        n_estimators = trial.suggest_int("rf_n_estimators", 25, 300)
        
        rf_model = RandomForestClassifier(
            max_depth=max_depth,
            max_samples=max_samples,
            n_estimators=n_estimators,
            random_state=42

        )
        score = cross_val_score(rf_model, X, Y, cv=5).mean()
        return score

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=50)

    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")
    #logging
    
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    return trial


if __name__ == "__main__":
    main()



