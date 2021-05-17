from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2

"""Deep Multimodal model"""


class DeepVDs:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def lrcn(self, custom_shape=None):
        if not custom_shape:
            input_layer = Input(shape=self.input_shape)
        else:
            input_layer = Input(shape=custom_shape)

        def add_default_block(X, kernel_filters, init, reg_lambda):
            X = TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same', kernel_initializer=init,
                                       kernel_regularizer=L2(l=reg_lambda)))(X)
            X = TimeDistributed(BatchNormalization())(X)
            X = TimeDistributed(Activation('relu'))(X)

            X = TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same', kernel_initializer=init,
                                       kernel_regularizer=L2(l=reg_lambda)))(X)
            # conv
            X = TimeDistributed(BatchNormalization())(X)
            X = TimeDistributed(Activation('relu'))(X)
            X = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(X)

            return X

        initialiser = 'glorot_uniform'
        reg_lambda = 0.001

        # first (non-default) block
        X = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                   kernel_initializer=initialiser, kernel_regularizer=L2(l=reg_lambda)))(input_layer)
        X = TimeDistributed(BatchNormalization())(X)
        X = TimeDistributed(Activation('relu'))(X)
        X = TimeDistributed(Conv2D(32, (3, 3), kernel_initializer=initialiser, kernel_regularizer=L2(l=reg_lambda)))(X)
        X = TimeDistributed(BatchNormalization())(X)
        X = TimeDistributed(Activation('relu'))(X)
        X = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(X)

        # 2nd-5th (default) blocks
        X = add_default_block(X, 64, init=initialiser, reg_lambda=reg_lambda)
        X = add_default_block(X, 128, init=initialiser, reg_lambda=reg_lambda)
        X = add_default_block(X, 256, init=initialiser, reg_lambda=reg_lambda)
        X = add_default_block(X, 512, init=initialiser, reg_lambda=reg_lambda)
        # X = add_default_block(X, 1024, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head
        X = TimeDistributed(Flatten())(X)
        X = LSTM(256, return_sequences=False, dropout=0.5)(X)

        return input_layer, X

    def get_conv_3d(self, custom_shape=None):
        if not custom_shape:
            input_layer = Input(shape=self.input_shape)
        else:
            input_layer = Input(shape=custom_shape)

        X = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
                   kernel_regularizer=L2(l=0.01))(input_layer)
        X = MaxPooling3D(pool_size=(2, 2, 2))(X)
        X = BatchNormalization()(X)
        # X = Dropout(0.2)(X)

        X = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
                   kernel_regularizer=L2(l=0.01))(X)
        X = MaxPooling3D(pool_size=(2, 2, 2))(X)
        X = BatchNormalization()(X)
        # X = Dropout(0.2)(X)

        X = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=L2(l=0.01),
                   kernel_initializer='he_uniform')(X)
        X = MaxPooling3D(pool_size=(2, 2, 2))(X)
        X = BatchNormalization()(X)
        # X = Dropout(0.5)(X)

        X = Conv3D(256, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=L2(l=0.01),
                   kernel_initializer='he_uniform')(X)
        X = MaxPooling3D(pool_size=(2, 2, 2))(X)
        X = BatchNormalization()(X)
        # X = Dropout(0.5)(X)

        X = Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=L2(l=0.01),
                   kernel_initializer='he_uniform')(X)
        X = MaxPooling3D(pool_size=(2, 2, 2))(X)
        X = BatchNormalization()(X)
        # X = Dropout(0.5)(X)

        fatten_layer = Flatten()(X)
        return input_layer, fatten_layer

    # TODO: Use transformers or try it
    def get_lstm(self, custom_shape=None):
        input_layer = Input(shape=custom_shape)
        X = LSTM(64, return_sequences=True)(input_layer)
        X = Dropout(0.4)(X)
        X = BatchNormalization()(X)
        X = LSTM(128)(X)
        X = Dropout(0.5)(X)
        X = Dense(units=256, activation='relu')(X)
        X = BatchNormalization()(X)
        fatten_layer = Flatten()(X)
        return input_layer, fatten_layer

    def conv_lstm(self, custom_shape=None):
        input_layer = Input(shape=custom_shape)
        X = TimeDistributed(Conv1D(filters=64, kernel_size=4, activation='relu'))(input_layer)
        X = TimeDistributed(Conv1D(filters=128, kernel_size=4, activation='relu'))(X)
        # X = TimeDistributed(MaxPooling1D(pool_size=2))(X)
        # X = TimeDistributed(Conv1D(filters=256, kernel_size=4, activation='relu'))(X)
        # X = Dropout(0.4)(X)
        X = TimeDistributed(MaxPooling1D(pool_size=2))(X)
        X = TimeDistributed(Flatten())(X)
        X = LSTM(64, recurrent_dropout=0.2)(X)
        # X = Dense(units=32, activation='relu')(X)
        X = BatchNormalization()(X)
        # fatten_layer = Flatten()(X)
        return input_layer, X

    def get_fully_connected_layers(self, merged_layers, classification=False):
        X = Dense(256, activation='relu', kernel_initializer='he_uniform')(merged_layers)
        X = Dropout(0.4)(X)
        X = BatchNormalization()(X)
        if not classification:
            final_layer = Dense(self.output_shape)(X)  # Linear Activation
        else:
            final_layer = Dense(self.output_shape, activation='softmax')(X)
        return final_layer

    def get_model(self, input_list, layer_list, classification):
        merge_layer = concatenate(layer_list)
        output = self.get_fully_connected_layers(merged_layers=merge_layer, classification=classification)
        model = Model(inputs=input_list, outputs=output)
        return model
