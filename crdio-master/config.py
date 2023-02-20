RAW_PATH = r"data/raw/cardio_data.csv"
TRAIN_PATH = r"data/raw/train_data.csv"
TEST_PATH = r"data/raw/test_data.csv"
PROCESSED_TRAIN_PATH = r"data/processed/processed_train.csv"
PROCESSED_TEST_PATH =r"data/processed/processed_test.csv"
TRAIN_FEATURES_PATH = r"data/processed/train_features.csv"
TEST_FEATURES_PATH = r"data/processed/test_features.csv"
ALL_FEATURES_TRAIN_PATH = r"data/processed/all_features_train.csv"
ALL_FEATURES_TEST_PATH= r"data/processed/all_features_test.csv"

MODEL_PATH = r"models/all_features_random_forest_trained_model.sav"
SCALER_PATH = r"models/std_scaler.sav"

INPUT_PATH = r"data/raw/test_input.csv"

SCORE_REPORT_PATH = r"reports/rand_forest_classificaiton_report.csv"
PREDICTION_PATH = r"reports/cardio_or_not_results.csv"


#training params

RANDOM_STATE = 42
N_ESTIMATORS = 120
MAX_DEPTH = 9
MAX_SAMPLES = 0.36
CRITERION = 'gini'