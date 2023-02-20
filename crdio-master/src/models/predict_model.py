import sys
import os
import pickle
import logging
import seaborn as sns
import pandas as pd
import numpy as np


from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.features.preprocess import preprocess
from src.features.build_features import feature_engineering
from src.features.prepare_modelling import prepare_modelling
import configparser


def show_conf_matrix(y_test, Y_hat):
    plt.rcParams['figure.figsize'] = (8, 8)
    sns_plot = sns.heatmap(confusion_matrix(
        y_test, Y_hat), annot=True, linewidths=.5, cmap="YlGnBu")
    fig = sns_plot.get_figure()
    plt.title('RANDOM FOREST CONFUSION MATRIX')
    fig.savefig("reports/figures/result_conf/rand_forest_conf_matrix.png")

def main():
    # get paths from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    processed_paths = config["PROCESSED"]
    model_paths = config["MODEL"]
    report_paths = config["REPORT"]


    np.random.seed(42)
    logging.basicConfig(level=logging.INFO,
                        filename="crdio/pred_logs.txt", filemode='w')

    df = pd.read_csv(processed_paths["ALL_FEATURES_TEST_PATH"], sep=";")

    X = df.drop(['cardio','id'], axis=1)
    Y = df['cardio']

    trained_model = pickle.load(open(model_paths["MODEL_PATH"], 'rb'))
    scaler_model = pickle.load(open(model_paths["SCALER_PATH"], 'rb'))

    # normalize, prediction, score
    X = scaler_model.transform(X)
    score = trained_model.score(X, Y)
    Y_hat = trained_model.predict(X)

    report = classification_report(Y, Y_hat, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(report_paths["SCORE_REPORT_PATH"],sep=";")
    logging.info("Score:"+str(score))

    #Confusion matrix
    show_conf_matrix(Y, Y_hat)
    print("model score : "+str(score))
    print('True Positive Cases : {}'.format(confusion_matrix(Y, Y_hat)[1][1]))
    print('True Negative Cases : {}'.format(confusion_matrix(Y, Y_hat)[0][0]))
    print('False Positive Cases : {}'.format(confusion_matrix(Y, Y_hat)[0][1]))
    print('False Negative Cases : {}'.format(confusion_matrix(Y, Y_hat)[1][0]))
    logging.info("----------------------------------------------------")
    logging.info("Confusion Matrix")
    logging.info('True Positive Cases : {}'.format(confusion_matrix(Y, Y_hat)[1][1]))
    logging.info('True Negative Cases : {}'.format(confusion_matrix(Y, Y_hat)[0][0]))
    logging.info('False Positive Cases : {}'.format(confusion_matrix(Y, Y_hat)[0][1]))
    logging.info('False Negative Cases : {}'.format(confusion_matrix(Y, Y_hat)[1][0]))


if __name__ == "__main__":
    main()
