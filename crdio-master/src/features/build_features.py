import numpy as np
import pandas as pd
import configparser

def BMI(data):
    return int((data['weight']/((data['height']/100)**2)))

def pulse(data):
    return np.subtract(data['ap_hi'], data['ap_lo'])


def feature_engineering(df):

    # calculate body mass index
    df['bmi'] = df.apply(BMI, axis=1)

    # calculate pulse
    df['pulse'] = df.apply(pulse, axis=1)

    # create categorical age feature
    df.loc[(df["age"] < 18), "new_age"] = "young"
    df.loc[(df["age"] >= 18), "new_age"] = "middle"
    df.loc[(df["age"] >= 54), "new_age"] = "old"

    # create body health feature by using bmi
    df.loc[(df["bmi"] <= 18), "new_bmi"] = "under"
    df.loc[(df["bmi"] > 18) & (df["bmi"] <= 25), "new_bmi"] = "healthy"
    df.loc[(df["bmi"] > 25) & (df["bmi"] <= 30), "new_bmi"] = "over"
    df.loc[(df["bmi"] > 30), "new_bmi"] = "obese"

    # create bloodpressure feature using ap_hi,ap_lo
    df.loc[(df["ap_lo"] <= 89), "blood_pressure"] = "normal"
    df.loc[(df["ap_lo"] >= 90), "blood_pressure"] = "hyper"
    df.loc[(df["ap_hi"] <= 120), "blood_pressure"] = "normal"
    df.loc[(df["ap_hi"] > 120), "blood_pressure"] = "middle"
    df.loc[(df["ap_hi"] >= 140), "blood_pressure"] = "hyper"

    # do labelencoding on string type categorical features    
    df["new_age"] = df['new_age'].map({'young':0, 'middle': 1, 'old':2})
    df["new_bmi"] = df['new_bmi'].map({'under':0, 'healthy': 1, 'over':2,'obese':3})
    df["blood_pressure"] = df['blood_pressure'].map({'normal':0, 'middle': 1, 'hyper':2})

    all_features_df = df.drop(["smoke","alco","active"],axis=1)

    # age,ap_hi,ap_lo,cholestrol and pulse features are selected
    new_df = df.drop(['bmi', 'weight', 'height', 'gluc', 'gender', 'smoke',
                     'alco', 'active', 'new_age', 'new_bmi', 'blood_pressure'], axis=1)

    return new_df,all_features_df


def main(): 
    # get paths from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    processed_paths = config["PROCESSED"]

    np.random.seed(42)
    processed_train_data = pd.read_csv(
        processed_paths["PROCESSED_TRAIN_PATH"], sep=";")
    processed_test_data = pd.read_csv(
        processed_paths["PROCESSED_TEST_PATH"], sep=";")

    train_features,all_train_features = feature_engineering(processed_train_data)
    test_features,all_test_features = feature_engineering(processed_test_data)

    #dataset before dropping features
    all_train_features.to_csv(processed_paths["ALL_FEATURES_TRAIN_PATH"], sep=";",index=False)
    all_test_features.to_csv(processed_paths["ALL_FEATURES_TEST_PATH"], sep=";",index=False)

    #dataset after dropping features
    train_features.to_csv(processed_paths["TRAIN_FEATURES_PATH"], sep=";",index=False)
    test_features.to_csv(processed_paths["TEST_FEATURES_PATH"], sep=";",index=False)
    return


if __name__ == '__main__':
    main()
