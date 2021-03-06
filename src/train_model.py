import string
import random
import sys
import pickle
import pathlib

from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# sys.path.append(pathlib.Path(__file__).parent)
from dataset import *


def create_callbacks(name_weights, patience_lr=10, patience_es=10):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    # return [early_stopping, mcp_save, reduce_lr_loss]
    return [early_stopping, mcp_save]


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def next_dataset(raw_data: RawData, batch_size: int,
                 is_train: bool = True, detect_wakeword: bool = False, feature_type=FEATURE_TYPE):
    """ Obtain a batch of training data
    """
    while True:
        train_dataset_x = []
        train_dataset_y = []
        for i in range(batch_size):
            # extract background
            bg_idx = np.random.randint(len(raw_data.backgrounds))
            background = raw_data.backgrounds[bg_idx]
            background_audio_data = background['audio']
            bg_segment = get_random_time_segment_bg(background_audio_data)
            clipped_background = background_audio_data[bg_segment[0]:bg_segment[1]]
            tmp_filename = f"sample_{''.join(random.choices(string.ascii_uppercase + string.digits, k=20))}.wav"
            x, y = create_training_sample(clipped_background, raw_data,
                                          filename=tmp_filename, is_train=is_train,
                                          feature_type=feature_type)
            # x, y = create_training_sample(clipped_background, raw_data, is_train=is_train)

            if detect_wakeword:
                y = np.delete(y, np.where(y == [0.0]), axis=0)
                if not len(y):
                    y = [0.0]
                else:
                    y = y[0]
            # print(y)
            train_dataset_x.append(x)
            train_dataset_y.append(y)

        # print(f"input:{np.array(train_dataset_x).shape} y_data:{np.array(train_dataset_y).shape}")
        yield np.array(train_dataset_x), np.array(train_dataset_y)


def train_model(model: Model, raw_data: RawData, model_filename,
                feature_type=FEATURE_TYPE,
                batch_size=100,
                num_train_examples=50000,
                num_valid_samples=2500,
                epochs=15,
                detect_wakeword=False):
    callbacks = create_callbacks(model_filename)
    steps_per_epoch = num_train_examples//batch_size
    validation_steps = num_valid_samples//batch_size
    # model.fit(x, y, validation_split=0.3, batch_size=batch_size, epochs=10, callbacks=callbacks)
    model.fit_generator(generator=next_dataset(raw_data, batch_size, is_train=True,
                                               detect_wakeword=detect_wakeword, feature_type=feature_type),
                        epochs=epochs,
                        validation_data=next_dataset(raw_data, batch_size, is_train=False,
                                                     detect_wakeword=detect_wakeword, feature_type=feature_type),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks, verbose=True)
