"""
dnn.py

Copyright (C) <2023>  Giuseppe Marco Randazzo <gmrandazzo@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import time
import os
from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import(
    Sequential,
    load_model
)
from keras.layers import(
    Dense,
    Dropout,
    BatchNormalization
)


from .dataset import Split

def tf_dnn(split : Split):
    """
    Create a simple feedforward neural network using TensorFlow.

    Parameters:
        split (Split): A `Split` object containing training, testing, and validation data splits.

    Returns:
        tensorflow.keras.models.Model: A TensorFlow Keras model representing the neural network.

    This function creates a simple feedforward neural network model using the TensorFlow library.
    The network can be used for tasks such as regression or classification. It takes as input
    a `Split` object containing the necessary data splits for training, testing, and validation.

    Example:
        # Create a Split object containing data splits
        data_split = make_split(xdict, x_header, ydict, random_state=42)

        # Create a neural network model and fit everything
        model = tf_dnn(data_split)
    """
    K.clear_session()
    print(">> Compute DNN with tensorflow")
    nfeatures = split.x_train.shape[1]
    # TUNING PARAMETERS
    nunits = nfeatures
    ndense_layers = 2
    epochs_= 3000
    batch_size_ = 20

    model = Sequential()
    model.add(BatchNormalization(input_shape=(nfeatures,)))
    model.add(Dense(nunits, activation='relu'))
    model.add(Dropout(0.15))
    for _ in range(ndense_layers):
        model.add(Dense(nunits, activation='relu'))
        model.add(Dropout(0.1))
    model.add(Dense(nunits, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=0.00005),
                  metrics=['mse', 'mae'])
    log_dir_ = './dnnlogs/'+time.strftime('%Y%m%d%H%M%S')
    model_output = "dnnmodel.h5"
    # Use model checkpoints to save the best predictive model
    callbacks_ = [TensorBoard(log_dir=log_dir_,
                              histogram_freq=0,
                              write_graph=False,
                              write_images=False),
                 ModelCheckpoint(model_output,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)]
    model.fit(split.x_train, split.y_train,
              epochs=epochs_,
              batch_size=batch_size_,
              verbose=1,
              validation_data=(split.x_test, split.y_test),
              callbacks=callbacks_)
    # Load the best model and predict the validation set
    bestmodel = load_model(model_output)
    y_val_pred = bestmodel.predict(split.x_val)
    os.remove(model_output)
    del model
    return y_val_pred
