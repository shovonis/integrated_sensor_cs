from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2


class Neural:
    def __init__(self, input_shape=None, output_shape=None):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_lstm(self, input_shape):
        if not input_shape:
            print("Error No input shape defined")
            return
        input_layer = Input(shape=input_shape)
        X = LSTM(60, return_sequences=True)(input_layer)
        X = LSTM(120, recurrent_dropout=0.2)(X)
        output = Dense(256, activation='relu')(X)
        return input_layer, output

    def conv_lstm(self, input_shape=None):
        input_layer = Input(shape=input_shape)
        X = TimeDistributed(Conv1D(filters=60, kernel_size=4, activation='relu'))(input_layer)
        X = TimeDistributed(Conv1D(filters=120, kernel_size=4, activation='relu'))(X)
        X = TimeDistributed(MaxPooling1D(pool_size=2))(X)
        # X = TimeDistributed(Conv1D(filters=256, kernel_size=4, activation='relu'))(X)
        X = Dropout(0.4)(X)
        # X = TimeDistributed(MaxPooling1D(pool_size=2))(X)
        X = TimeDistributed(Flatten())(X)
        X = LSTM(120, recurrent_dropout=0.2)(X)
        X = Dense(units=256, activation='relu')(X)
        X = BatchNormalization()(X)
        fatten_layer = Flatten()(X)
        return input_layer, fatten_layer

    def get_regression_model(self, input_layers, output_layers, merge=True):
        if merge:
            merge_layer = concatenate(output_layers)
        else:
            merge_layer = output_layers

        final = Dense(1)(merge_layer)
        model = Model(inputs=input_layers, outputs=final)
        return model

    def get_classification_model(self, input_layers, output_layers, merge=True):
        if merge:
            merge_layer = concatenate(output_layers)
        else:
            merge_layer = output_layers[0]

        final = Dense(self.output_shape, activation='softmax')(merge_layer)
        model = Model(inputs=input_layers, outputs=final)
        return model
