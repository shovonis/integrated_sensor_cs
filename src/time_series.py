from neural import model as ml
from data_processor import data_processor as dp
import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from result_generation import result_generator
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def print_and_save_model(model):
    print(model.summary())
    logging.info("See model summary at: .../results/")
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    summary_file = '../results/model_summary-' + current_time + '.txt'
    with open(summary_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    if hyper_parameters["classification"]:
        file_name = '../model/model_classification' + current_time + '.h5'
    else:
        file_name = '../model/model_regression' + current_time + '.h5'
    model.save(file_name)


def do_k_fold_evaluation(model, eye_X, head_X, target, fold=10):
    current_fold = 1
    kfold = StratifiedKFold(n_splits=fold, shuffle=True)
    list_acc = []
    list_loss = []
    list_history = []
    list_precisions = []
    list_recall = []
    list_f1 = []
    list_r2 = []
    list_plcc = []
    list_srcc = []
    for train, test in kfold.split(eye_X, target):
        print("### Train on Fold: ", current_fold)

        history = model.fit(x=[eye_X[train], head_X[train]], y=target[train], epochs=hyper_parameters["Epochs"],
                            batch_size=hyper_parameters["batch_size"], validation_split=0.2, verbose=1, shuffle=False)

        list_history.append(history)

        print("\n ### Evaluate (Test data) on Fold : ", current_fold)
        # TODO: Add or remove modalities here

        if hyper_parameters["classification"]:
            loss, accuracy = model.evaluate(x=[eye_X[test], head_X[test]], y=target[test], batch_size=64, verbose=1)
            print('On Fold %d test loss: %.3f' % (current_fold, loss))
            list_loss.append(loss)  # Loss is universal

            # Make predictions
            actual_cs = target[test]  # Ground Truth
            predicted_cs = model.predict(x=[eye_X[test], head_X[test]])

            print("Actual CS: ", actual_cs)
            print('On Fold %d test accuracy: %.3f' % (current_fold, accuracy))
            list_acc.append(accuracy)
            predicted_cs = np.argmax(predicted_cs, axis=1)
            precisions, recall, f1_score = result_generator.process_results_for_classification(
                predicted_cs=predicted_cs, actual_cs=actual_cs, output_shape=hyper_parameters["number_of_class"])
            list_precisions.append(precisions)
            list_recall.append(recall)
            list_f1.append(f1_score)
        else:
            loss = model.evaluate(x=[eye_X[test], head_X[test]], y=target[test], batch_size=64, verbose=1)
            print('On Fold %d test loss: %.3f' % (current_fold, loss[0]))
            list_loss.append(loss[0])  # Loss is universal

            # Make predictions
            actual_cs = target[test]  # Ground Truth
            predicted_cs = model.predict(x=[eye_X[test], head_X[test]])

            print("Actual CS: ", actual_cs)
            predicted_cs = list(np.concatenate(predicted_cs).flat)
            print("Predicted CS: ", predicted_cs)

            r2 = r2_score(actual_cs, predicted_cs, multioutput='variance_weighted')
            list_r2.append(r2)
            print('On Fold %d test R2: %.3f' % (current_fold, r2))

            plcc, _ = pearsonr(actual_cs, predicted_cs)
            print('On Fold %d test PLCC: %.3f' % (current_fold, plcc))
            list_plcc.append(plcc)

            srcc, _ = spearmanr(actual_cs, predicted_cs)
            print('On Fold %d test SRCC: %.3f' % (current_fold, srcc))
            list_srcc.append(srcc)

        current_fold += 1

    print("### K-Fold Result Summary...")

    if hyper_parameters["classification"]:
        print("K-Fold Mean ACC and STD ACC", np.mean(list_acc), np.std(list_acc))
        print("K- Fold Mean Loss and STD Loss", np.mean(list_loss), np.std(list_loss))

        print("K-Fold Total Precision ")
        print(list_precisions)

        print("K-Fold Total Recall ")
        print(list_recall)

        print("K-Fold Total F1_score ")
        print(list_f1)
    else:
        print("K- Fold Mean Loss and STD Loss", np.mean(list_loss), np.std(list_loss))
        print("K- Fold Mean R2 and STD R2", np.mean(list_r2), np.std(list_r2))
        print("K- Fold Mean PLCC and STD PLCC", np.mean(list_plcc), np.std(list_plcc))
        print("K- Fold Mean SRCC and STD SRCC", np.mean(list_srcc), np.std(list_srcc))


def train_model():
    data_processor = dp.DataProcessor(classification=hyper_parameters["classification"])
    if hyper_parameters["classification"]:
        model = ml.Neural(output_shape=hyper_parameters["number_of_class"])
    else:
        model = ml.Neural(output_shape=1)

    # Eye Data
    print("Processing Eye Tracking data")
    eye_data = data_processor.get_data_from_file(modalities_paths["Eye"])
    eye_data = data_processor.prepare_time_series_data(eye_data, time_step=hyper_parameters["time_step"],
                                                       output_dim=1)
    eye_data = eye_data[eye_data.columns[~eye_data.columns.to_series().str.contains(pat='X1\(')]]
    # eye_data.to_csv("test.csv")

    eye_X, eye_Y = data_processor.get_x_y_data(data=eye_data, time_step=hyper_parameters["time_step"],
                                               number_of_features=hyper_parameters["eye_features"])

    # eye_X,  eye_Y = manage_imbalance_class(eye_X, eye_Y)
    print("Eye Data X Shape: ", eye_X.shape)
    print("Eye Data Y Shape: ", eye_Y.shape)

    eye_input_layer, eye_output_layer = model.get_lstm(input_shape=(eye_X.shape[1], eye_X.shape[2]))

    # Head Data
    print("Processing Head Tracking data")
    head_data = data_processor.get_data_from_file(modalities_paths["Head"])
    head_data = data_processor.prepare_time_series_data(head_data, time_step=hyper_parameters["time_step"],
                                                        output_dim=1)

    head_data = head_data[head_data.columns[~head_data.columns.to_series().str.contains(pat='X1\(')]]

    head_X, head_Y = data_processor.get_x_y_data(data=head_data, time_step=hyper_parameters["time_step"],
                                                 number_of_features=hyper_parameters["head_features"])

    # head_X,  head_Y = manage_imbalance_class(head_X, head_Y)

    print("Head Data X Shape: ", head_X.shape)
    print("Head Data Y Shape: ", head_Y.shape)

    head_input_layer, head_output_layer = model.get_lstm(input_shape=(head_X.shape[1], head_X.shape[2]))

    if hyper_parameters["classification"]:
        # Get Model and Train
        model = model.get_classification_model(input_layers=[eye_input_layer, head_input_layer],
                                               output_layers=[eye_output_layer, head_output_layer],
                                               merge=True)
        # Compile and Train Model TODO: Do it for Regression
        target = eye_Y  # Set it to head or eye target both are same
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    else:
        model = model.get_regression_model(input_layers=[eye_input_layer, head_input_layer],
                                           output_layers=[eye_output_layer, head_output_layer],
                                           merge=True)
        target = eye_Y  # Set it to head or eye target both are same
        model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mae', 'mse'])

        print(target)

        # Print and Save model
    print_and_save_model(model)

    # K Fold Cross validation
    fold = 10
    do_k_fold_evaluation(model=model, eye_X=eye_X, head_X=head_X, target=target, fold=fold)


if __name__ == '__main__':
    # Setup TF Constants
    seed_constant = 11
    np.random.seed(seed_constant)
    np.random.seed(seed_constant)
    tf.random.set_seed(seed_constant)
    tf.get_logger().setLevel('INFO')

    logging.basicConfig(filename='../log/server.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    hyper_parameters = {"time_step": 60, "eye_features": 9, "head_features": 4, "Epochs": 10,
                        "classification": True, "number_of_class": 4, "concatenate": False, "batch_size": 512}

    modalities = {"Eye": True, "Head": True, "Clips": False, "Optic": False, "Disparity": False}

    modalities_paths = {"Eye": '../data2/eye/', "Head": '../data2/head/'}

    # Train The model
    train_model()
