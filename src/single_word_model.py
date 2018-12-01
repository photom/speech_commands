import sys
import pathlib

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, \
    Conv3D, Conv1D, Flatten, MaxPooling1D, MaxPooling3D
from keras.layers import GRU, Bidirectional, BatchNormalization
from keras.layers import Input, ELU
from keras.layers import Reshape
from keras.optimizers import Adam

sys.path.append(pathlib.Path(__file__).parent)
from dataset import *
import command


def elu(x, alpha=0.05):
    return K.elu(x, alpha)


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


def create_model_cnn0(input_shape, class_num=command.NUM_CLASSES,
                      dropout=0.5):
    """
    https://arxiv.org/pdf/1803.03759.pdf
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance

    # 5layers: loss: 0.1482 - acc: 0.9581 - val_loss: 0.5575 - val_acc: 0.8728
    """
    x_input = Input(name='the_input', shape=input_shape)
    x = Reshape((Tx, 28, 28, 3))(x_input)
    # CONV layer
    x = Conv3D(filters=32, kernel_size=(5, 5, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=0.05)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), dim_ordering='tf')(x)

    # CONV layer
    for i in range(2):
        x = Conv3D(filters=64, kernel_size=(5, 5, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=0.05)(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), dim_ordering='tf')(x)
    for i in range(2):
        x = Conv3D(filters=128, kernel_size=(5, 5, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=0.05)(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1), dim_ordering='tf')(x)

    # Batch normalization added right before the ReLu activation
    x = Dense(1024, activation=elu)(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=x)

    return model


def create_model_cnn(input_shape, class_num=command.NUM_CLASSES,
                     dropout=0.55, is_train=True):
    """
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43969.pdf
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance

    # 3layers, no pool: loss: 0.0811 - acc: 0.9795 - val_loss: 0.1311 - val_acc: 0.9588
    # 3layers           loss: 0.1453 - acc: 0.9560 - val_loss: 0.1152 - val_acc: 0.9636
    # 5layers           loss: 0.1263 - acc: 0.9621 - val_loss: 0.1317 - val_acc: 0.9624
    # 4layers           loss: 0.0870 - acc: 0.9752 - val_loss: 0.1010 - val_acc: 0.9712
    # 4layers  no dilation loss: 0.0914 - acc: 0.9745 - val_loss: 0.1155 - val_acc: 0.9688
    """

    x_input = Input(name='the_input', shape=input_shape)
    x = Reshape((Tx, 40))(x_input)
    # CONV layer
    x = Conv1D(filters=32, kernel_size=(10,), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=0.05)(x)
    x = MaxPooling1D(pool_size=(5,), strides=(2,))(x)

    for i in range(1):
        x = Conv1D(filters=64, kernel_size=(5,), padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=0.05)(x)
        x = MaxPooling1D(pool_size=(3,), strides=(2,))(x)

    for i in range(1):
        x = Conv1D(filters=128, kernel_size=(5,), padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=0.05)(x)
        x = MaxPooling1D(pool_size=(3,), strides=(2,))(x)

    for i in range(1):
        x = Conv1D(filters=256, kernel_size=(3,), padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=0.05)(x)
        x = MaxPooling1D(pool_size=(2,), strides=(2,))(x)

    # Batch normalization added right before the ReLu activation
    x = Dense(1024, activation=elu)(x)
    if is_train:
        x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=x)
    # model.output_length = lambda v: cnn_output_length(v, filter_size=196, border_mode='causal',
    #                                                   stride=1, dilation=2)
    return model


def build_model(model_filename, learning_rate=0.001,
                create_model=create_model_cnn,
                input_shape=(Tx, n_freq)):
    print(f"input shape={(Tx, n_freq)}")
    model = create_model(input_shape=(Tx, n_freq), class_num=command.NUM_CLASSES)
    if os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)

    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model
