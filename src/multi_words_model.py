from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, \
    Conv3D, Conv1D, Flatten, MaxPooling1D, MaxPooling3D
from keras.layers import GRU, Bidirectional, BatchNormalization
from keras.layers import Input, ELU, Lambda
from keras.layers import Reshape
from keras.optimizers import Adam
from keras import backend as keras_backend

from dataset import *
import command


def elu(x, alpha=0.05):
    return K.elu(x, alpha)


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


def create_model_cousera(input_shape, class_num=command.NUM_CLASSES):
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
    x = TimeDistributed(Dense(class_num, activation="softmax"))(x)  # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs=x_input, outputs=x)

    return model


def create_model_dilation(input_shape, class_num=command.NUM_CLASSES,
                          units=200, filters=200, kernel_size=11,
                          conv_layers=1,
                          dropout=0.5, recurrent_dropout=0.02,
                          is_train=True):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    kmn_dilation2.weights.best.hdf5: val_loss: 0.2173 - val_acc: 0.927
    kmn_dilation3.weights.best.hdf5: val_loss: 0.1992 - val_acc: 0.934
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
    # time distributed  (sigmoid)
    x = TimeDistributed(Dense(class_num, activation="softmax"))(x)

    model = Model(inputs=x_input, outputs=x)
    model.output_length = lambda v: cnn_output_length(v, kernel_size, 'causal',
                                                      stride=1, dilation=2)
    return model


def create_model_cnn_bidirect(input_shape, class_num=command.NUM_CLASSES,
                              units=200, filters=196, kernel_size=11,
                              dropout=0.5, recurrent_dropout=0.1,
                              is_train=True):
    """
    Function creating the model's graph in Keras.
    loss: 0.1503 - acc: 0.9493 - val_loss: 0.1244 - val_acc: 0.9575
    loss: 0.1600 - acc: 0.9455 - val_loss: 0.1258 - val_acc: 0.9560
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(name='the_input', shape=input_shape)
    x = x_input
    # CONV layer
    x = Conv1D(filters=filters, kernel_size=(5,), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=0.05)(x)
    x = MaxPooling1D(pool_size=(5,), strides=(2,))(x)
    # if is_train:
    #    x = Dropout(dropout)(x)

    for i in range(2):
        x = Bidirectional(GRU(units=units,
                              return_sequences=True,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              implementation=1))(x)
        x = BatchNormalization()(x)

    # Time-distributed dense layer
    x = TimeDistributed(Dense(class_num, activation='softmax'))(x)

    model = Model(inputs=x_input, outputs=x)
    model.output_length = lambda v: cnn_output_length(v, kernel_size, 'causal',
                                                      stride=2, dilation=2)
    return model


def create_model_cnn_bidirect_ctc(input_shape, class_num=command.NUM_CLASSES,
                                  units=200, filters=196, kernel_size=11,
                                  dropout=0.5, recurrent_dropout=0.1,
                                  is_train=True):
    model = create_model_cnn_bidirect(input_shape, class_num, units, filters, kernel_size, dropout, recurrent_dropout,
                                      is_train)
    return add_ctc_loss(model)


def create_model_bidirect(input_shape, class_num=command.NUM_CLASSES,
                          units=200, filters=196, kernel_size=11,
                          dropout=0.5, recurrent_dropout=0.1,
                          is_train=True):
    """
    Function creating the model's graph in Keras.
    loss: 0.1721 - acc: 0.9437 - val_loss: 0.1576 - val_acc: 0.9452

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(name='the_input', shape=input_shape)
    x = x_input

    for i in range(2):
        x = Bidirectional(GRU(units=units,
                              return_sequences=True,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              implementation=1))(x)
        x = BatchNormalization()(x)
    # if is_train:
    #    x = Dropout(dropout)(x)

    # Time-distributed dense layer
    x = TimeDistributed(Dense(class_num, activation='softmax'))(x)

    model = Model(inputs=x_input, outputs=x)
    model.output_length = lambda v: cnn_output_length(v, kernel_size, 'causal',
                                                      stride=2, dilation=2)
    return model


def create_model_strides(input_shape, class_num=command.NUM_CLASSES,
                         units=200, filters=200, kernel_size=11,
                         conv_layers=2, recur_layers=2,
                         dropout=0.5, recurrent_dropout=0.2):
    """
    Function creating the model's graph in Keras.
    loss: 0.1770 - acc: 0.9422 - val_loss: 0.1295 - val_acc: 0.9552

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
                              recurrent_dropout=recurrent_dropout,
                              implementation=1))(x)
        x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    # Time-distributed dense layer (≈1 line)
    x = TimeDistributed(Dense(class_num, activation="softmax"))(x)

    model = Model(inputs=x_input, outputs=x)
    return model


def build_model(model_filename, learning_rate=0.001,
                create_model=create_model_dilation,
                input_shape=(Tx, n_freq)):
    print(f"input shape={(Tx, n_freq)}")
    model = create_model(input_shape=(Tx, n_freq), class_num=command.NUM_CLASSES)
    if os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)

    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model
