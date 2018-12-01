#!/usr/bin/env python

import pickle
import random
import string

from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataset import *
from multi_words_model import *


def create_callbacks(name_weights, patience_lr=5, patience_es=5):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4,
                                       mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    # return [early_stopping, mcp_save, reduce_lr_loss]
    return [early_stopping, mcp_save]


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def next_dataset_ctc(raw_data: RawData, batch_size, is_train=True, feature_type=FEATURE_TYPE):
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
            background = raw_data.backgrounds[np.random.randint(len(raw_data.backgrounds))]
            background_audio_data = background['audio']
            bg_segment = get_random_time_segment_bg(background_audio_data)
            clipped_background = background_audio_data[bg_segment[0]:bg_segment[1]]
            tmp_filename = f"sample_{''.join(random.choices(string.ascii_uppercase + string.digits, k=20))}.wav"
            # add keywords
            x, y = create_training_sample(clipped_background, raw_data, filename=tmp_filename,
                                          is_train=is_train, feature_type=feature_type)

            # calculate X_data & input_length
            input_length[i] = x.shape[1]
            # input_length[i] = 969
            x_data[i, :x.shape[0], :] = x
            # set label
            words = []
            for word in y:
                if word[0] != 0 and word[0] not in words:
                    words.append(word[0])
            labels.append(np.array(words))

        # calculate labels & label_length
        max_string_length = max(WORD_NUM_RATIO)
        y_data = np.ones([batch_size, max_string_length]) * command.NUM_CLASSES
        # print(f"y_data:{y_data.shape} labels:{len(labels)} label:{labels[0].shape},{labels[0][:, 0].shape} max_string_length:{max_string_length}")
        for i in range(batch_size):
            label = labels[i]
            y_data[i, :len(label)] = label
            label_length[i] = len(label)
        # print(f"input_length:{input_length}")
        # print(f"label_length:{label_length}")
        # print(f"labels:{labels}")
        print(f"input:{x_data.shape} y_data:{y_data.shape} inputlen:{input_length.shape} label_len:{label_length.shape}")
        # return the arrays
        outputs = {'ctc': np.zeros([batch_size])}
        inputs = {'the_input': x_data,
                  'the_labels': y_data,
                  'input_length': input_length,
                  'label_length': label_length, }
        yield inputs, outputs


def train_model_ctc(model: Model, raw_data: RawData, model_filename,
                    feature_type=FEATURE_TYPE,
                    batch_size=100,
                    num_train_examples=50000,
                    num_valid_samples=2500,
                    epochs=15, ):
    callbacks = create_callbacks(model_filename)
    steps_per_epoch = num_train_examples // batch_size
    validation_steps = num_valid_samples // batch_size
    # model.fit(x, y, validation_split=0.3, batch_size=batch_size, epochs=10, callbacks=callbacks)
    model.fit_generator(
        generator=next_dataset_ctc(raw_data, batch_size, is_train=True, feature_type=feature_type),
        epochs=epochs,
        validation_data=next_dataset_ctc(raw_data, batch_size, is_train=False, feature_type=feature_type),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=True)


def build_model_ctc(model_filename, create_model=create_model_cnn_bidirect_ctc):
    model = create_model(input_shape=(Tx, n_freq))
    if os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)

    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    return model
