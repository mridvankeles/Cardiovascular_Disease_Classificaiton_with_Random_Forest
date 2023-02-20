import logging
import pandas as pd
import pickle
import numpy as np

from src.features.preprocess import preprocess
from src.features.build_features import feature_engineering
import configparser

def main():
    # get paths from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    test_path = config["TEST"]
    model_paths = config["MODEL"]
    report_paths = config["REPORT"]


    np.random.seed(42)
    # Read input data and type check
    missing_values = ['?', '--', ' ', 'NA', 'N/A',
                      '-', 'null', 'Null', 'nill', 'None']
    df = pd.read_csv(test_path["INPUT_PATH"], sep=";",na_values=missing_values)
    if type(df) != pd.DataFrame:
        df = pd.DataFrame(df)

    # Apply preprocess and feature extraction to input data
    processed_data = preprocess(df,"test")
    features_df,all_features_df = feature_engineering(processed_data)

    # Get only features for model prediction and load models
    X = all_features_df.drop(['id'], axis=1)
    trained_model = pickle.load(open(model_paths["MODEL_PATH"], 'rb'))
    scaler_model = pickle.load(open(model_paths["SCALER_PATH"], 'rb'))
    
    # Normalize input and make prediction
    X = scaler_model.transform(X)
    Y_hat = trained_model.predict(X)
    pred=pd.DataFrame({"id":all_features_df["id"],"cardio_pred":Y_hat})
    pred.to_csv(report_paths["PREDICTION_PATH"],sep=";",index=False)


if __name__ == "__main__":
    main()