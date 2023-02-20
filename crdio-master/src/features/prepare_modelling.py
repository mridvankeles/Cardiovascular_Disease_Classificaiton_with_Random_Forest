import sys
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import configparser

sys.path.append(os.getcwd())


def prepare_modelling(df):

    config = configparser.ConfigParser()
    config.read("config.ini")
    proc_data = config["PROCESSED"]

    X = df.drop(['id', 'cardio'], axis=1)
    Y = df['cardio']



    scaler = StandardScaler()
    X=scaler.fit_transform(X)

    




    return X, Y, scaler
