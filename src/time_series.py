from neural import model as ml
from src.data_processor import data_processor as dp
import tensorflow as tf
import numpy as np
import logging


def print_and_save_model(model):
    print(model.summary())
    logging.info("See model summary at: .../results/model_summary.txt")
    with open('../results/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    if hyper_parameters["classification"]:
        model.save('../model/model_classification.h5')
    else:
        model.save('../model/model_regression.h5')


def train_model():
    data_processor = dp.DataProcessor(classification=hyper_parameters["classification"])
    model = ml.Neural(output_shape=hyper_parameters["number_of_class"])
    history = []

    # Eye Data
    print("Processing Eye Tracking data")
    eye_data = data_processor.get_data_from_file(modalities_paths["Eye"])
    eye_data = data_processor.prepare_time_series_data(eye_data, time_step=hyper_parameters["time_step"],
                                                       output_dim=1)
    eye_X, eye_Y = data_processor.get_x_y_data(data=eye_data, time_step=hyper_parameters["time_step"],
                                               number_of_features=hyper_parameters["eye_features"])
    print("Eye Data X Shape: ", eye_X.shape)
    print("Eye Data Y Shape: ", eye_Y.shape)

    eye_input_layer, eye_output_layer = model.get_lstm(input_shape=(eye_X.shape[1], eye_X.shape[2]))

    # Head Data
    print("Processing Head Tracking data")
    head_data = data_processor.get_data_from_file(modalities_paths["Head"])
    head_data = data_processor.prepare_time_series_data(head_data, time_step=hyper_parameters["time_step"],
                                                        output_dim=1)
    head_X, head_Y = data_processor.get_x_y_data(data=head_data, time_step=hyper_parameters["time_step"],
                                                 number_of_features=hyper_parameters["head_features"])
    print("Head Data X Shape: ", head_X.shape)
    print("Head Data Y Shape: ", head_Y.shape)

    head_input_layer, head_output_layer = model.get_lstm(input_shape=(head_X.shape[1], head_X.shape[2]))

    # Get Model and Train
    model = model.get_classification_model(input_layers=[eye_input_layer, head_input_layer],
                                           output_layers=[eye_output_layer, head_output_layer],
                                           merge=True)

    # Compile and Train Model TODO: Do it for Regression
    target = eye_Y  # Set it to head or eye target both are same
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print and Save model
    print_and_save_model(model)

    history = model.fit(x=[eye_X, head_X], y=target, epochs=50, batch_size=512, validation_split=0.2, verbose=1,
                        shuffle=False)


if __name__ == '__main__':
    # Setup TF Constants
    seed_constant = 11
    np.random.seed(seed_constant)
    np.random.seed(seed_constant)
    tf.random.set_seed(seed_constant)
    tf.get_logger().setLevel('INFO')

    logging.basicConfig(filename='../log/server.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    hyper_parameters = {"time_step": 60, "eye_features": 9, "head_features": 4, "Epochs": 50,
                        "classification": True, "number_of_class": 3, "concatenate": False}
    modalities = {"Eye": True, "Head": False, "Clips": False, "Optic": False, "Disparity": False}
    modalities_paths = {"Eye": '../data/eye/', "Head": '../data/head/'}

    # Train The model
    train_model()
