from cmath import log
import os
import sys
import argparse
import optparse
import logging
import datetime


from src.data import make_dataset
from src.models import train_model,predict_model,predict_output

from src.visualization import visualize
from src.features import preprocess
from src.features import build_features
from src.models import optimize


logging.basicConfig(level = logging.INFO, filename="logs.txt", filemode = 'a')
argparser = argparse.ArgumentParser()

argparser.add_argument("--train", help="Runs only the training part. Requires training data.\n"
    "Usage : python runproject.py --train True",)
argparser.add_argument("--data", help="Creates only necessary data files.\n"
    "Usage : python runproject.py --data True")
argparser.add_argument("--viz", help= "Saves the prepared plots to reports/ folder.\n"
    "Usage : python runproject.py --viz True")
argparser.add_argument("--prep", help = "Makes preprocessing operations.\n"
    "Usage : python runproject.py --prep True")
argparser.add_argument("--features", help = "Makes feature operations and drops unnecessary columns..\n"
    "Usage : python runproject.py --features True")
argparser.add_argument("--optim", help = "Optimizes the hyperparameters of given classifier..\n"
    "Usage : python runproject.py --optim True")
argparser.add_argument("--predict", help = "Makes predictions with the test data.\n"
    "Usage : python runproject.py --predict True")
argparser.add_argument("--all", help = "To execute the whole project use this option.\n"
    "Usage : python runproject.py --all True")
argparser.add_argument("--input", help = "Takes an input path and returns predictions.")

args = argparser.parse_args()

sys.path.append(os.getcwd())


if __name__ == "__main__":
    if bool(args.train) == True:
        train_model.main()
        logging.info("Training at " +  str(datetime.datetime.now()) +  " has been executed.")
    if bool(args.data) == True:
        make_dataset.main()
    if bool(args.prep) == True:
        preprocess.main()
    if bool(args.features) == True:
        build_features.main()
    if bool(args.viz) == True:
        visualize.main()
    if bool(args.predict) == True:
        predict_model.main()
    if bool(args.optim) == True:
        optimize.main()
    if bool(args.input) == True:
        predict_output.main()
    if bool(args.all) == True:
        make_dataset.main()
        print("dataset created.")
        preprocess.main()
        print("preproces done.")
        build_features.main()
        print("features extracted.")
        visualize.main()
        print("visualizations saved.")
        train_model.main()
        print("training done.")
        predict_model.main()
        print("prediction saved.")
        logging.info("All project has been executed at " +  str(datetime.datetime.now()) + ".")
        logging.info("-------------------------------------------------------------------")
    
    
    

    








