# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import pandas as pd
import sys
import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import configparser
# config dosyaso oluştur


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logging.basicConfig(level=logging.INFO,
                        filename="crdio/logs.txt", filemode='a')

    # get paths from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    data_paths = config["DATASET"]

    const_seed = np.random.seed(42)
    df = pd.read_csv(data_paths["RAW_PATH"],sep=";")
    train, test = train_test_split(df, test_size=0.1, random_state=const_seed)
    logging.info("Train shape: " + str(train.shape) +
                 "\n" + "Test shape: " + str(test.shape))
    test.to_csv(data_paths["TEST_PATH"], sep=";", index=False)
    train.to_csv(data_paths["TRAIN_PATH"], sep=";", index=False)

if __name__ == '__main__':
    main()
