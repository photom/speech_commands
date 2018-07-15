#!/usr/bin/env python

import sys
import time
import random
import string
import shutil
from functools import partial
from multiprocessing.dummy import Pool
import threading
from io import BytesIO
import traceback
from collections import Counter
from timeit import default_timer as timer

import numpy as np
from speech_recognition import Microphone
from speech_recognition import Recognizer
from speech_recognition import AudioData
from keras.models import Model

from kr_model import *
import noisered


SAMPLE_RATE = 16000
PHRASE_TIME_LIMIT = 2
MODEL_WEIGHT_PATH = 'model/kmn_dilation2.weights.best.hdf5'

THREAD_NUM = 1
# SHARED_MEM_DIR = f"/dev/shm/keyword_recognizer_{''.join(random.choices(string.ascii_uppercase + string.digits, k=10))}"
SHARED_MEM_DIR = "/var/tmp/keyword_recognizer"

PROFILE = 'profile'
INPUT_WAV = 'input.wav'
NOISERED_WAV = 'noisered.wav'
BG_WAV = 'bg_input.wav'

PROFILE_PATH = os.path.join(SHARED_MEM_DIR, PROFILE)
INPUT_WAV_PATH = os.path.join(SHARED_MEM_DIR, INPUT_WAV)
NOISERED_WAV_PATH = os.path.join(SHARED_MEM_DIR, NOISERED_WAV)
BG_WAV_PATH = os.path.join(SHARED_MEM_DIR, BG_WAV)


class ModelMap(object):
    def __init__(self):
        self.models = {}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def initialize_shared_mem_dir(dir_name=SHARED_MEM_DIR):
    if os.path.exists(dir_name):
        eprint('temp dir already exists in shared mem.')
        exit(1)
    else:
        os.makedirs(dir_name)


def remove_shared_mem_dir(dir_name=SHARED_MEM_DIR):
    try:
        shutil.rmtree(dir_name)
    except OSError as e:
        eprint("Error: %s - %s." % (e.filename, e.strerror))


def load_model():
    model = create_model_dilation(input_shape=(Tx, n_freq), is_train=False)
    model.load_weights('model/kmn_dilation2.weights.best.hdf5')
    model.summary()
    return model


def summarize_prediction(predicted):
    decoded = [np.argmax(predicted[idx], axis=0) for idx, key_idx in enumerate(predicted)]
    print(dict(Counter(decoded)))


def predict_word(audio_data: AudioData, model_map: ModelMap):
    try:
        if not os.path.exists(PROFILE_PATH):
            print("noise profile is not ready.")
            return
        try:
            os.remove(INPUT_WAV_PATH)
        except:
            pass

        # execute noise reduction
        with open(INPUT_WAV_PATH, 'wb') as f:
            f.write(audio_data.get_wav_data())
        noisered.create_noisered_wav(INPUT_WAV_PATH, NOISERED_WAV_PATH, PROFILE_PATH)

        # load or get model
        if threading.get_ident() not in model_map.models:
            print(f"load model. tid:{threading.get_ident()}")
            model_map.models[threading.get_ident()] = load_model()
        model = model_map.models[threading.get_ident()]

        # create input from wav data
        # io_obj = BytesIO(audio_data.get_wav_data())
        # x = create_mfcc_from_io(io_obj)
        x = create_mfcc_from_file(NOISERED_WAV_PATH)
        # x = create_mfcc_from_file(INPUT_WAV_PATH)

        # complement shortage space
        print(f"x:{x.shape},{x.dtype} framedata:{len(audio_data.frame_data)}")
        if x.shape[0] < Tx:
            min_val = np.amin(x, axis=0)
            print(f"min_val:{min_val.shape}")
            # calc remaining space size
            empty_space_size = Tx - x.shape[0]
            # create remaining space
            empty_space = np.tile(min_val, (empty_space_size, 1))
            # empty_space = np.zeros((empty_space_size, n_freq), dtype=np.float32)
            # complement data's empty space
            print(f"min_val:{min_val.shape} emptysp:{empty_space.shape}")
            x = np.concatenate((x, empty_space), axis=0)
        # frames = np.array(data)
        if x.shape[0] > Tx:
            eprint(f"trim input. from={x.shape[0]} to={Tx}")
            x = x[:Tx]
        x = np.float32(np.array([x]))
        print(f"x:{x.shape},{x.dtype}")

        # do predict
        start = timer()
        predicted = model.predict(x)
        end = timer()
        print(f"predicted: {end - start}")
        summarize_prediction(predicted[0])
    except:
        traceback.print_exc()
        raise


def callback(_: Recognizer, audio_data: AudioData, model_map: ModelMap, pool: Pool):
    pool.apply_async(predict_word, (audio_data, model_map,))


def listen_background():
    background_listener = noisered.BackgroundListener()
    with Microphone() as source:
        background_listener.adjust_for_ambient_noise(source)
        while os.path.exists(SHARED_MEM_DIR):
            audio_data = background_listener.record(source, duration=3)
            try:
                os.remove(BG_WAV_PATH)
            except:
                pass

            # create wav file
            with open(BG_WAV_PATH, 'wb') as f:
                f.write(audio_data.get_wav_data())
            # create profile with sox
            noisered.create_noiseprof(BG_WAV_PATH, PROFILE_PATH)


def start_listen_background():
    t = threading.Thread(target=listen_background, name='listen_background')
    t.start()


def main():
    # start to listen background with another thread.
    start_listen_background()

    while not os.path.exists(PROFILE_PATH):
        print('ready for noise profile ...')
        time.sleep(1)

    # initialize recognizer
    recognizer = Recognizer()
    with Microphone(sample_rate=SAMPLE_RATE) as source:
        # listen for 1 second to calibrate the energy threshold for ambient noise levels
        recognizer.adjust_for_ambient_noise(source)
        print("Calibrated. Say something!")

    source = Microphone(sample_rate=SAMPLE_RATE)
    with Pool(THREAD_NUM) as pool:
        callback_with_model = partial(callback, model_map=ModelMap(), pool=pool)
        recognizer.listen_in_background(source, callback_with_model, PHRASE_TIME_LIMIT)
        while True:
            time.sleep(1)
        # pool.terminate()


if __name__ == '__main__':
    try:
        # initialize_shared_mem_dir()
        main()
    finally:
        pass
        # remove_shared_mem_dir()
