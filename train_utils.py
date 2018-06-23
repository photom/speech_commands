#!/usr/bin/env python

import numpy as np
from td_utils import *

import pickle

from keras.models import Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import set_random_seed
from keras.layers import (Input, Lambda)
from keras import backend as keras_backend


def cnn_output_length(input_length, filter_size, border_mode, stride,
                      dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same`, `valid` or 'causal'.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif border_mode == 'causal':
        # as code below, causal mode's output_length is input_length.
        # https://github.com/fchollet/keras/blob/master/keras/utils/conv_utils.py#L112
        output_length = input_length
    else:
        raise RuntimeError(f"invalid border_mode={border_mode}")
    return (output_length + stride - 1) // stride


def create_model_cousera(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    x_input = Input(name='the_input', shape=input_shape)

    ### START CODE HERE ###

    # Step 1: CONV layer (≈4 lines)
    x = Conv1D(filters=196, kernel_size=15, strides=4)(x_input)  # CONV1D
    x = BatchNormalization()(x)  # Batch normalization
    x = Activation("relu")(x)  # ReLu activation
    x = Dropout(0.8)(x)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    x = GRU(units=128, return_sequences=True)(x)  # GRU (use 128 units and return the sequences)
    x = Dropout(0.8)(x)  # dropout (use 0.8)
    x = BatchNormalization()(x)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    x = GRU(units=128, return_sequences=True)(x)  # GRU (use 128 units and return the sequences)
    x = Dropout(0.8)(x)  # dropout (use 0.8)
    x = BatchNormalization()(x)  # Batch normalization
    x = Dropout(0.8)(x)  # dropout (use 0.8)
    # x = Reshape((Ty*(128/NUM_CLASSES), NUM_CLASSES))(x)
    # Step 4: Time-distributed dense layer (≈1 line)
    x = TimeDistributed(Dense(NUM_CLASSES, activation="softmax"))(x)  # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs=x_input, outputs=x)

    return model


def create_model_dilation(input_shape, units=200, filters=200, kernel_size=11,
                          conv_layers=1,
                          dropout=0.5, recurrent_dropout=0.02,
                          is_train=True):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    if not is_train:
        dropout = 0
        recurrent_dropout = 0

    x_input = Input(name='the_input', shape=input_shape)
    x = x_input
    for i in range(conv_layers):
        x = Conv1D(filters=filters, kernel_size=kernel_size,
                   dilation_rate=2 ** i,
                   name=f"conv1d{i}")(x)
        x = BatchNormalization(name=f"bn_conv1d{i}")(x)
        x = Activation("relu")(x)
        if is_train:
            x = Dropout(dropout)(x)

    x = Bidirectional(GRU(units=units,
                          return_sequences=True,
                          recurrent_dropout=recurrent_dropout,
                          dropout=dropout,
                          implementation=1,
                          name=f"bidirectional_gru1"))(x)
    x = BatchNormalization()(x)  # Batch normalization
    if is_train:
        x = Dropout(dropout)(x)

    # Step 4: Time-distributed dense layer (≈1 line)
    x = TimeDistributed(Dense(NUM_CLASSES, activation="softmax"))(x)  # time distributed  (sigmoid)

    model = Model(inputs=x_input, outputs=x)
    model.output_length = lambda v: cnn_output_length(v, kernel_size, 'causal',
                                                      stride=1, dilation=2)
    return model


def create_model_strides(input_shape, units=200, filters=200, kernel_size=11,
                         conv_layers=2, recur_layers=2,
                         dropout=0.5, recurrent_dropout=0.1):
    """
    Function creating the model's graph in Keras.
    val_loss: 0.0963 - val_acc: 0.912
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    x_input = Input(name='the_input', shape=input_shape)
    x = x_input
    for i in range(conv_layers):
        x = Conv1D(filters=filters, kernel_size=kernel_size,
                   strides=2 ** i,
                   name=f"conv1d{i}")(x)
        x = BatchNormalization(name=f"bn_conv1d{i}")(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)

    for i in range(recur_layers):
        x = Bidirectional(GRU(units=units,
                              return_sequences=True,
                              dropout=dropout,
                              implementation=1,
                              name=f"gru{i}"))(x)
        x = BatchNormalization()(x)  # Batch normalization
    x = Dropout(0.5)(x)

    # Step 4: Time-distributed dense layer (≈1 line)
    x = TimeDistributed(Dense(NUM_CLASSES, activation="softmax"))(x)  # time distributed  (sigmoid)

    model = Model(inputs=x_input, outputs=x)
    return model


def create_callbacks(name_weights, patience_lr=10, patience_es=5):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    return [early_stopping, mcp_save, reduce_lr_loss]
    # return [early_stopping, mcp_save]


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def next_dataset(raw_data: RawData, batch_size: int, is_train: bool = True):
    """ Obtain a batch of training data
    """
    while True:
        train_dataset_x = []
        train_dataset_y = []
        for i in range(batch_size):
            # extract background
            background = raw_data.backgrounds[np.random.randint(len(raw_data.backgrounds))]
            background_audio_data = background['audio']
            bg_segment = get_random_time_segment_bg(background_audio_data)
            clipped_background = background_audio_data[bg_segment[0]:bg_segment[1]]

            x, y = create_training_sample(clipped_background, raw_data, is_train=is_train)
            train_dataset_x.append(x)
            train_dataset_y.append(y)

        # print(f"input:{np.array(train_dataset_x).shape} y_data:{np.array(train_dataset_y).shape}")
        yield np.array(train_dataset_x), np.array(train_dataset_y)


def train_model(model: Model, raw_data: RawData, model_filename,
                batch_size=100,
                num_train_examples=10000,
                num_valid_samples=500,
                epochs=20):
    callbacks = create_callbacks(model_filename)
    steps_per_epoch = num_train_examples//batch_size
    validation_steps = num_valid_samples//batch_size
    # model.fit(x, y, validation_split=0.3, batch_size=batch_size, epochs=10, callbacks=callbacks)
    model.fit_generator(generator=next_dataset(raw_data, batch_size, is_train=True),
                        epochs=epochs,
                        validation_data=next_dataset(raw_data, batch_size, is_train=False),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks, verbose=True)


def build_model(model_filename, learning_rate=0.001, create_model=create_model_dilation_1):
    model = create_model(input_shape=(Tx, n_freq))
    if os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)

    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model


def build_and_train(raw_data, model_filename, create_model=create_model_dilation_1):
    model = build_model(model_filename, learning_rate=0.001, create_model=create_model)
    train_model(model, raw_data, model_filename,
                num_train_examples=10000,
                num_valid_samples=500)

