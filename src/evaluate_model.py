import datetime
import logging
import random
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
import result_generation.result_generator as rg
from data_processor.data_generator import DataGenerator
from neural import multi_modal_nn as ml
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def manage_imbalance_class(meta_data):
    class1, class2, class3 = meta_data['cs_class'].value_counts()
    logging.info("Current Distribution of Class")
    logging.info("Class 1:  %s, Class 2: %s, Class 3: %s", class1, class2, class3)
    logging.info("Random Oversampling class")
    X = meta_data
    y = meta_data['cs_class']
    ros = RandomOverSampler(random_state=42)
    X_res, _ = ros.fit_resample(X, y)
    return X_res


def evaluate_model():
    model = load_model('../model/deep_mcs_model_regression.h5')
    print(model.summary())

    input_shape = (300, 64, 64, 3)  # Shape (time_step, H, W, D)
    time_step_for_time_series = 300
    eye_features = 9
    head_features = 4
    batch_size = 4
    epochs = 500
    classification = False
    base_path = '../data'

    meta_data = pd.read_csv('../data/meta_data.csv')
    meta_data = manage_imbalance_class(meta_data)
    _, test = train_test_split(meta_data, test_size=0.2)
    test = test.reset_index(drop=True)
    test_generator = DataGenerator(dataset=test, base_path=base_path, image_shape=(input_shape[1], input_shape[2]),
                                   time_step_image=input_shape[0], batch_size=1,
                                   classification=classification, time_step_time_series=time_step_for_time_series,
                                   shuffle=False)

    # print("Evaluate the Model on test data...")
    # model.evaluate(test_generator, verbose=1)

    predicted_cs = model.predict(test_generator)
    actual_cs = test['fms'].to_numpy()
    predicted_cs = list(np.concatenate(predicted_cs).flat)

    print("Predicted: ", predicted_cs )
    print("Actual: ", actual_cs)

    r2 = r2_score(actual_cs, predicted_cs, multioutput='variance_weighted')
    print("R2: ", r2)

    plcc, _ = pearsonr(actual_cs, predicted_cs)
    print("PLCC: ", plcc)
    # sorc, _ = spearmanr(actual_cs, predicted_cs)
    # print(f'PLCC : {0} and SORC {1}', plcc, sorc)


evaluate_model()
