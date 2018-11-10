#!/usr/bin/env python

from src.td_utils_ctc import *

import pickle
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Input, Lambda)
from keras import backend as keras_backend


def cnn_output_length(input_length, filter_size, border_mode, stride=1,
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


def create_model_ctc(input_shape, units=200, filters=200, kernel_size=11,
                     conv_layers=2, recur_layers=2,
                     dropout=0.8, recurrent_dropout=0.1,
                     is_train=False,
                     have_ctc=True):
    """
    Function creating the model's graph in Keras.
    val_loss: 0.0478 - val_acc: 0.908
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    if not is_train:
        dropout = 0

    x_input = Input(name='the_input', shape=input_shape)
    x = x_input
    for i in range(conv_layers):
        x = Conv1D(filters=filters, kernel_size=kernel_size,
                   dilation_rate=2 ** i,
                   # strides=2 ** i,
                   name=f"conv1d{i}")(x)
        x = BatchNormalization(name=f"bn_conv1d{i}")(x)
        x = Activation("relu")(x)
        if is_train:
            x = Dropout(dropout)(x)

    for i in range(recur_layers):
        x = Bidirectional(GRU(units=units,
                              return_sequences=True,
                              dropout=dropout,
                              implementation=1,
                              name=f"gru{i}"))(x)
        x = BatchNormalization()(x)  # Batch normalization
    if is_train:
        x = Dropout(dropout)(x)

    # Step 4: Time-distributed dense layer (≈1 line)
    x = TimeDistributed(Dense(len(char_map.CHAR_MAP), activation="softmax"))(x)

    model = Model(inputs=x_input, outputs=x)
    model.output_length = lambda v: cnn_output_length(v, kernel_size, 'causal',
                                                      stride=1, dilation=2)
    # model.output_length = lambda x: x
    if have_ctc:
        # add CTC loss to the NN specified in input_to_softmax
        model = add_ctc_loss(model)

    return model


def create_callbacks(name_weights, patience_lr=5, patience_es=5):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    # return [early_stopping, mcp_save, reduce_lr_loss]
    return [early_stopping, mcp_save]


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # print(f"ctc_lambda_func:: input:{y_pred.shape} label:{labels.shape} inputlen:{input_length.shape} label_len:{label_length.shape}")
    return keras_backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths],
        outputs=loss_out)
    return model


def next_dataset_ctc(backgrounds, keywords, batch_size, is_train=True):
    """ Obtain a batch of training data
    """
    while True:
        # initialize the arrays
        x_data = np.zeros([batch_size, Tx, n_freq])
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        # print(f"x_data: {x_data.shape}")
        # print(f"labels: {labels.shape}")

        # fill in data
        labels = []
        for i in range(batch_size):
            # extract background
            background = backgrounds[np.random.randint(len(backgrounds))]
            background_audio_data = background['audio']
            bg_segment = get_random_time_segment_bg(background_audio_data)
            clipped_background = background_audio_data[bg_segment[0]:bg_segment[1]]
            # add keywords
            feat, label = create_training_sample(clipped_background, keywords, is_train=is_train)

            # calculate X_data & input_length
            # input_length[i] = feat.shape[0]
            input_length[i] = 969
            x_data[i, :feat.shape[0], :] = feat
            # set label
            labels.append(label)

        # calculate labels & label_length
        max_string_length = max([len(label) for label in labels])
        y_data = np.ones([batch_size, max_string_length]) * 28
        for i in range(batch_size):
            label = labels[i]
            y_data[i, :len(label)] = label
            label_length[i] = len(label)
        # print(f"input_length:{input_length}")
        # print(f"label_length:{label_length}")
        # print(f"labels:{labels}")
        # print(f"input:{x_data.shape} y_data:{y_data.shape} inputlen:{input_length.shape} label_len:{label_length.shape}")
        # return the arrays
        outputs = {'ctc': np.zeros([batch_size])}
        inputs = {'the_input': x_data,
                  'the_labels': y_data,
                  'input_length': input_length,
                  'label_length': label_length,}
        yield inputs, outputs


def train_model_ctc(model, backgrounds, keywords, model_filename,
                    batch_size=50,
                    num_train_examples=10000,
                    num_valid_samples=500,
                    epochs=20):
    callbacks = create_callbacks(model_filename)
    steps_per_epoch = num_train_examples//batch_size
    validation_steps = num_valid_samples//batch_size
    # model.fit(x, y, validation_split=0.3, batch_size=batch_size, epochs=10, callbacks=callbacks)
    model.fit_generator(generator=next_dataset_ctc(backgrounds, keywords, batch_size, is_train=True),
                        epochs=epochs,
                        validation_data=next_dataset_ctc(backgrounds, keywords, batch_size, is_train=False),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        verbose=True)


def build_model_ctc(model_filename, create_model=create_model_ctc):
    model = create_model(input_shape=(Tx, n_freq))
    if os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)

    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    return model
