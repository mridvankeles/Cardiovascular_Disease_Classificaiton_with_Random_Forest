import sys
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from src.features.prepare_modelling import prepare_modelling
import configparser

def main():
    # get paths from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    processed_paths = config["PROCESSED"]
    model_paths = config["MODEL"]


    seed = np.random.seed(42)
    print("training...")
    df = pd.read_csv(processed_paths["ALL_FEATURES_TRAIN_PATH"], sep=";")
    X, Y, scaler = prepare_modelling(df)
    X_train, X_val, y_train, y_val =  train_test_split(X,Y,test_size=0.33,random_state=42)

    if model_paths["MODEL_NAME"] == "rf":
        model = RandomForestClassifier(random_state=seed, n_estimators=120,
                                    max_depth=9, max_samples=0.36,
                                    criterion='gini')
        model.fit(X, Y)
        pickle.dump(model, open(model_paths["MODEL_PATH"], 'wb'))
        pickle.dump(scaler, open(model_paths["SCALER_PATH"], 'wb'))
        print("trained model saved.")


    elif model_paths["MODEL_NAME"] == "nn":
    
        def build_model(n_hidden=8,n_neurons=100,learning_rate=0.001,input_shape=[13,],reg_lambda=7e-05,activation_func="elu"):

            model = keras.models.Sequential()
            model.add(Dense(n_neurons, input_shape=input_shape,activation=activation_func,kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(reg_lambda)))
            for layer in range(n_hidden):
                model.add(Dense(n_neurons, activation=activation_func,kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(reg_lambda))),
                model.add(keras.layers.Dropout(0.2))
                
            model.add(keras.layers.Dense(1,activation="sigmoid",kernel_initializer="glorot_uniform"))
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
            loss_fn = keras.losses.BinaryCrossentropy()
            model.compile(loss=loss_fn, optimizer=optimizer,metrics = ['acc'])
            return model
        model_train = keras.wrappers.scikit_learn.KerasClassifier(build_model)


        def exponential_decay_fn(epoch):
            return 0.001

        lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
        

        history = model_train.fit(X_train, y_train, epochs=150,batch_size = 100,
            validation_data=(X_val, y_val),
            callbacks=[keras.callbacks.EarlyStopping(patience=20),lr_scheduler]
        )

        def show_model_scores(model):
            y_val_pred = model.predict(X_val)

            y_val_pred_=pd.DataFrame(y_val_pred)
            y_val_pred_.index = y_val.index
            compare_nn_pred = pd.concat([y_val,y_val_pred_.round(2)],axis=1)
            print(compare_nn_pred)
        
        show_model_scores(model_train)

        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
        plt.show()

        history.model.save(model_paths["NN_MODEL_PATH"])

if __name__ == "__main__":
    main()
