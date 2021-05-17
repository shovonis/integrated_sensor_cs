import pandas as pd
import os
import tensorflow as tf


class DataProcessor:
    def __init__(self, classification=True):
        self.classification = classification

    @staticmethod
    def prepare_time_series_data(data, time_step=1, output_dim=1, drop_na=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()

        for i in range(time_step, 0, -1):
            cols.append(df.shift(i))
            names += [('X%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

        for i in range(0, output_dim):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('Y%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('Y%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

        aggregated_data = pd.concat(cols, axis=1)
        aggregated_data.columns = names

        if drop_na:
            aggregated_data.dropna(inplace=True)
        return aggregated_data

    def get_data_from_file(self, file_path):
        files = os.listdir(file_path)
        data = pd.DataFrame()
        for file in files:
            tmp_data = pd.read_csv(file_path + file)
            if self.classification:
                first_column = tmp_data.pop('cs')
                tmp_data.drop(columns=['fms', '#Frame'], inplace=True)
                tmp_data.insert(0, 'cs', first_column)
                data = data.append(tmp_data, ignore_index=True)
            else:
                first_column = tmp_data.pop('fms')
                tmp_data.drop(columns=['cs', '#Frame'], inplace=True)
                tmp_data.insert(0, 'fms', first_column)
                data = data.append(tmp_data, ignore_index=True)

        return data

    def get_x_y_data(self, data, time_step, number_of_features):
        values = data.values
        number_observation = time_step * number_of_features
        X, Y = values[:, :number_observation], values[:, -(number_of_features + 1)]
        X = X.reshape((X.shape[0], time_step, number_of_features))
        if self.classification:
            Y = tf.keras.utils.to_categorical(Y, num_classes=3)

        return X, Y
