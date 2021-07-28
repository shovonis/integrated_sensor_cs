import datetime
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from data_processor.data_generator import DataGenerator
from neural import multi_modal_nn as ml


def _construct_model():
    multimodal = ml.DeepVDs(input_shape=input_shape, output_shape=output_shape)

    left_input_clip, left_flatten_layer = multimodal.lrcn(input_shape)
    # right_input_clip, right_flatten_layer = multimodal.get_conv_3d(input_shape)

    input_optic, flatten_optics = multimodal.lrcn(input_shape)
    input_disp, flatten_disp = multimodal.lrcn(input_shape)

    # input_eye, flatten_eye = multimodal.conv_lstm(
    #     custom_shape=(batch_size, time_step_for_time_series // batch_size, eye_features))
    # #
    # input_head, flatten_head = multimodal.conv_lstm(custom_shape=(batch_size, time_step_for_time_series // batch_size,
    #                                                               head_features))

    # Get the full model
    model = multimodal.get_model([left_input_clip, input_optic, input_disp],
                                 [left_flatten_layer, flatten_optics, flatten_disp],
                                 classification=classification)

    return model


def _get_data_generators(train, test):
    # Get the Data generator
    train_gen = DataGenerator(dataset=train, base_path=base_path, image_shape=(input_shape[1], input_shape[2]),
                              time_step_image=input_shape[0], batch_size=batch_size, classification=classification,
                              time_step_time_series=time_step_for_time_series)

    # Get the Validation generator
    validation_gen = DataGenerator(dataset=test, base_path=base_path, image_shape=(input_shape[1], input_shape[2]),
                                   time_step_image=input_shape[0], batch_size=batch_size,
                                   classification=classification, time_step_time_series=time_step_for_time_series,
                                   shuffle=False)

    return train_gen, validation_gen


def manage_imbalance_class(meta_data):
    # class1, class2, class3 = meta_data['cs_class'].value_counts()
    # logging.info("Current Distribution of Class")
    # logging.info("Class 1:  %s, Class 2: %s, Class 3: %s", class1, class2, class3)
    # logging.info("Random Oversampling class")
    X = meta_data
    y = meta_data['cs_class']
    ros = RandomOverSampler(random_state=42)
    X_res, _ = ros.fit_resample(X, y)
    return X_res

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def train_model():
    # Split the train and test data
    logging.info("Train Test Split....")
    train, test = train_test_split(meta_data, test_size=0.3)
    # Reindex them (Must do this)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    # # Get the generators
    train_gen, validation_gen = _get_data_generators(train, test)
    # # Compile Model and Fit model
    model = _construct_model()
    print(model.summary())
    logging.info("See model summary at: .../results/model_summary.txt")
    with open('../results/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # # Save Model
    if classification:
        model.save('../model/deep_mcs_model_classification.h5')
    else:
        model.save('../model/deep_mcs_model_regression.h5')

    # # Train the Model
    log_dir = "../log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping = callbacks.EarlyStopping(monitor="val_accuracy",
                                             mode="max", patience=10,
                                             restore_best_weights=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    train_history = model.fit(train_gen, validation_data=validation_gen, verbose=1, epochs=epochs,
                              callbacks=[tensorboard_callback, early_stopping])

    # # Evaluate Model on Test Data
    print("Evaluate the Model on test data...")
    model.evaluate(validation_gen, verbose=1)

    # # Prediction
    test_generator = DataGenerator(dataset=test, base_path=base_path, image_shape=(input_shape[1], input_shape[2]),
                                   time_step_image=input_shape[0], batch_size=1,
                                   classification=classification, time_step_time_series=input_shape[0],
                                   shuffle=False)

    print("Make Prediction on test data...")

    predicted_cs = model.predict(test_generator)
    actual_cs = test['fms'].to_numpy()
    predicted_cs = list(np.concatenate(predicted_cs).flat)
    print("Predicted: ", predicted_cs)
    print("Actual: ", actual_cs)
    r2 = r2_score(actual_cs, predicted_cs, multioutput='variance_weighted')
    print("R2: ", r2)
    plcc, _ = pearsonr(actual_cs, predicted_cs)
    print("PLCC: ", plcc)




if __name__ == "__main__":
    # Initiate Log file
    logging.basicConfig(filename='../log/server.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info("Starting Training the model")
    seed_constant = 11
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)

    # Data path
    base_path = '../data3'
    meta_data = pd.read_csv('../data3/meta_data.csv')
    logging.info("Check for imbalance class")
    meta_data = manage_imbalance_class(meta_data)
    logging.info("................Current Hyper Parameters.......................")
    input_shape = (60, 128, 128, 3)
    time_step_for_time_series = 60
    eye_features = 9
    head_features = 4
    batch_size = 4
    epochs = 50
    classification = True

    logging.info("Image Shape: %s", input_shape)
    logging.info("Eye data shape: %s", (input_shape[0], eye_features))
    logging.info("Head data shape: %s", (input_shape[0], head_features))
    logging.info("Epochs %s", epochs)
    logging.info("Batch Size %s", batch_size)

    logging.info("Doing classification....")
    output_shape = 4

    train_model()
