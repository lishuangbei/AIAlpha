import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import pandas as pd
import numpy as np


class AutoEncoder:
    def __init__(self, encoding_dim, input_shape):
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape

    def build_model(self, encoded1_shape, encoded2_shape, decoded1_shape, decoded2_shape):
        input_data = Input(shape=(1, self.input_shape))

        # encoded1 = Dense(encoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
        # encoded2 = Dense(encoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded1)
        # encoded3 = Dense(self.encoding_dim, activation="relu", activity_regularizer=regularizers.l2(0))(encoded2)
        # decoded1 = Dense(decoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded3)
        # decoded2 = Dense(decoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(decoded1)
        # decoded = Dense(self.input_shape, activation="sigmoid", activity_regularizer=regularizers.l2(0))(decoded2)

        encoded3 = Dense(self.encoding_dim, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
        decoded = Dense(self.input_shape, activation="sigmoid", activity_regularizer=regularizers.l2(0))(encoded3)

        self.autoencoder = Model(inputs=input_data, outputs=decoded)
        self.encoder = Model(input_data, encoded3)

    def train_model(self, model, data, epochs, model_name, save_model=True):

        model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])

        train = data
        ntrain = np.array(train)
        train_data = np.reshape(ntrain, (len(ntrain), 1, self.input_shape))

        model.fit(train_data, train_data, epochs=epochs)

        if save_model:
            model.save(f"models/saved_models/{model_name}.h5")
    
    def test_model(self, model, data):
        test = data
        ntest = np.array(test)
        test_data = np.reshape(ntest, (len(ntest), 1, self.input_shape))

        print(model.evaluate(test_data, test_data))
        
    def encode_data(self, data, csv_path, save_csv=True):
        coded_train = []
        using_for_loop = 0
        num_of_samples = len(data)
        if using_for_loop:
            # For me using for-loop takes hours and eats up all the memory. Keras doesn't recommend this as well.
            for i in range(len(data)): # data.shape = (192xxx, 184)
                curr_data = np.array(data.iloc[i, :]) # pick one sample, shape is (1, 184)
                values = np.reshape(curr_data, (1, 1, self.input_shape)) # shape is (1, 1, 184)
                coded = self.encoder.predict(values) # shape is (1, 1, 20)
                # can above 'predict()' process massive data instead of 1 row at a time?
                shaped = np.reshape(coded, (20,)) #shape is (20, )
                coded_train.append(shaped) #
                print(i)
            train_coded = pd.DataFrame(coded_train, index=np.arange(len(coded_train)), columns=np.arange(20))
        else:
            # The 'predict()' step takes some time. For people who're using VS Code, set "PYDEVD_WARN_EVALUATION_TIMEOUT" to a large number in lauch.json.
            curr_data = np.array(data)
            values = np.reshape(curr_data, (num_of_samples, 1, self.input_shape)) # not sure which shape is correct, (num, 1, 184) or (num, 184).
            coded = self.encoder.predict(values)
            shaped = np.reshape(coded, (num_of_samples, 20))
            train_coded = pd.DataFrame(shaped, index=np.arange(len(shaped)), columns=np.arange(20))

        if save_csv:
            train_coded.to_csv(f"{csv_path}")
        return train_coded
    