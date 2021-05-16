import datetime
import logging
import random
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
import result_generation.result_generator as rg
from data_processor.data_generator import DataGenerator
from neural import multi_modal_nn as ml


def _construct_model():
    multimodal = ml.DeepVDs(input_shape=input_shape, output_shape=output_shape)
    input_clip, flatten_clips = multimodal.get_conv_3d(input_shape)
    input_optic, flatten_optics = multimodal.get_conv_3d(input_shape)
    input_disp, flatten_disp = multimodal.get_conv_3d(input_shape)

    input_eye, flatten_eye = multimodal.get_lstm(custom_shape=(time_step_for_time_series, eye_features))
    input_head, flatten_head = multimodal.get_lstm(custom_shape=(time_step_for_time_series, head_features))

    # Get the full model
    model = multimodal.get_model([input_clip, input_optic, input_eye],
                                 [flatten_clips, flatten_optics, flatten_eye],
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
                                   shuffle=True)

    return train_gen, validation_gen


def manage_imbalance_class(meta_data):
    class1, class2, class3 = meta_data['cs_class'].value_counts()
    logging.info("Current Distribution of Class")
    logging.info("Class 1:  %s, Class 2: %s, Class 3: %s", class1, class2, class3)
    logging.info("Random Oversampling class")
    X = meta_data
    y = meta_data['fms']
    ros = RandomOverSampler(random_state=42)
    X_res, _ = ros.fit_resample(X, y)
    return X_res


def train_model():
    # Split the train and test data
    logging.info("Train Test Split....")
    train, test = train_test_split(meta_data, test_size=0.2)
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
                                             mode="max", patience=20,
                                             restore_best_weights=True)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
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
    actual_cs = test['cs_class'].to_numpy()
    predicted_cs = model.predict(test_generator)
    predicted_cs = np.argmax(predicted_cs, axis=1)
    confusion_matrix = mt.confusion_matrix(actual_cs, predicted_cs)
    rg.plot_confusion_matrix(confusion_matrix, classes=list(range(output_shape - 1)))
    print(confusion_matrix)

    precisions, recall, f1_score, _ = mt.precision_recall_fscore_support(actual_cs, predicted_cs,
                                                                         labels=[0, 1, 2])
    print("Precision: ")
    print(precisions)
    print("Recall: ")
    print(recall)
    print("F1: ")
    print(f1_score)
    # Train History
    rg.plot_train_vs_val_loss(train_history)
    rg.plot_train_vs_val_acc(train_history)


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
    base_path = '../data'
    meta_data = pd.read_csv('../data/meta_data.csv')
    logging.info("Check for imbalance class")
    meta_data = manage_imbalance_class(meta_data)
    logging.info("Data Shape after oversampling: %s", meta_data.shape)

    # Setup the Hyper Parameters
    logging.info("................Current Hyper Parameters.......................")
    input_shape = (128, 128, 128, 3)
    time_step_for_time_series = 128
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
    output_shape = 3

    train_model()
