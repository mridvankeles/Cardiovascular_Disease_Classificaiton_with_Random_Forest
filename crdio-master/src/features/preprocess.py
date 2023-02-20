from operator import index
from unicodedata import numeric
import pandas as pd
import logging
import numpy as np
from sklearn.impute import SimpleImputer
import configparser

def check_missing(df):
    features_with_null = [
        feature for feature in df.columns if df[feature].isnull().sum() > 0]
    if features_with_null:
        return True
    else:

        return False


def check_duplicate(df):
    duplicate_sum = df.duplicated().sum()
    if duplicate_sum:
        return True
    else:
        return False


def drop_outliers(df, hw_max, hw_min, hilo_max, hilo_min):
    df.drop(df[(df['height'] > df['height'].quantile(hw_max)) | (
        df['height'] < df['height'].quantile(hw_min))].index, inplace=True)
    df.drop(df[(df['weight'] > df['weight'].quantile(hw_max)) | (
        df['weight'] < df['weight'].quantile(hw_min))].index, inplace=True)
    df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(hilo_max)) | (
        df['ap_hi'] < df['ap_hi'].quantile(hilo_min))].index, inplace=True)
    df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(hilo_max)) | (
        df['ap_lo'] < df['ap_lo'].quantile(hilo_min))].index, inplace=True)
    return df


#label encoder, train testte random seed, 
def Impute_Missing(df, data):
    imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    categorical_df = df[['gender', 'cholesterol',
                         'gluc', 'smoke', 'alco', 'active']]
    numerical_df = df[['age', 'weight', 'height', 'ap_hi', 'ap_lo']]
    if data == "train":
        imputer_num.fit_transform(numerical_df)
        imputer_cat.fit_transform(categorical_df)
    elif data == "test":
        imputer_num.fit_transform(numerical_df)
        imputer_cat.fit_transform(categorical_df)
    df = pd.concat([numerical_df, categorical_df, df['cardio']])
    return df


def preprocess(df, data):
    logging.basicConfig(level=logging.INFO,filename="crdio/logs.txt", filemode='a')
    # Impute missing values if exists
    if check_missing(df):
        df = Impute_Missing(df, data)
        logging.info("Missing values filled..")
        logging.info(df.head())
    else:
        logging.info("There are no missing values in any features.")

    # Drop outliers for numerical features
    if data == "train":
         # Drop duplicate values if exist
        if check_duplicate(df):
            df.drop_duplicates(keep='first', inplace=True)

        # Change age values to year type
        df['age'] = (df['age'] / 365).round().astype('int')

        # After age value change check and drop duplicates again
        if check_duplicate(df):
            df.drop_duplicates(keep='first', inplace=True)

        df = drop_outliers(df, hw_max=0.997, hw_min=0.03,
                           hilo_max=0.975, hilo_min=0.025)
    elif data == "test":
        # Change age values to year type
        df['age'] = (df['age'] / 365).round().astype('int')

        # !!Do not drop duplicates or outliers due to test data loss

    return df


def main():
    # get paths from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    data_paths = config["DATASET"]
    processed_paths = config["PROCESSED"]

    np.random.seed(42)
    logging.basicConfig(level=logging.INFO,
                        filename="crdio/logs.txt", filemode='a')

    missing_values = ['?', '--', ' ', 'NA', 'N/A',
                      '-', 'null', 'Null', 'nill', 'None']

    test = pd.read_csv(data_paths["TEST_PATH"],
                       sep=";", na_values=missing_values)
    train = pd.read_csv(data_paths["TRAIN_PATH"],
                        sep=";", na_values=missing_values)

    processed_train_data = preprocess(train, "train")
    processed_test_data = preprocess(test, "test")

    processed_train_data.to_csv(
        processed_paths["PROCESSED_TRAIN_PATH"], sep=";", index=False)
    processed_test_data.to_csv(
        processed_paths["PROCESSED_TEST_PATH"], sep=";", index=False)
    return


if __name__ == '__main__':
    main()
