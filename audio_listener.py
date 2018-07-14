#!/usr/bin/env python

from functools import partial

from speech_recognition import Microphone
from speech_recognition import Recognizer
from speech_recognition import AudioData
from keras.models import Model

from kr_model import *
import numpy as np


PHRASE_TIME_LIMIT = 2
MODEL_WEIGHT_PATH = 'model/kmn_dilation2.weights.best.hdf5'


def load_model():
    model = create_model_dilation(input_shape=(Tx, n_freq), is_train=False)
    model.load_weights('model/kmn_dilation2.weights.best.hdf5')
    return model


def callback(model: Model, recognizer: Recognizer, audio_data: AudioData):
    frames = []
    data = np.fromstring(audio_data.frame_data, dtype=np.int16)
    frames.append(data)
    frames = np.array(frames)
    print(frames.shape)
    #predicted = model.predict(frames)
    # print(model.metrics_names)
    #print(predicted)


def main():
    model = load_model()
    callback_with_model = partial(callback, model=model)
    recognizer = Recognizer()

    with Microphone(sample_rate=16000) as source:
        # listen for 1 second to calibrate the energy threshold for ambient noise levels
        recognizer.adjust_for_ambient_noise(source)
        print("Say something!")

        recognizer.listen_in_background(source, callback_with_model, PHRASE_TIME_LIMIT)


if __name__ == '__main__':
    main()
