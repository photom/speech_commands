#!/usr/bin/env python

import os
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
from typing import Union

from pydub import AudioSegment
import numpy as np
from speech_recognition import Microphone
from speech_recognition import Recognizer
from speech_recognition import AudioData
from keras.models import Model

from kr_model import *
import noisered


SAMPLE_RATE = 16000
PHRASE_TIME_LIMIT = 2
MODEL_WEIGHT_PATH = 'model/kmn_dilation_lbfe.weights.best.hdf5'

THREAD_NUM = 1
# SHARED_MEM_DIR = f"/dev/shm/keyword_recognizer_{''.join(random.choices(string.ascii_uppercase + string.digits, k=10))}"
SHARED_MEM_DIR = "/var/tmp/keyword_recognizer"

INPUT_WAV = 'input.wav'
NOISERED_WAV = 'noisered.wav'
BG_WAV = 'bg_input.wav'

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
        if not os.path.exists(BG_WAV_PATH):
            print("bg audio is not ready.")
            return
        try:
            os.remove(INPUT_WAV_PATH)
        except:
            pass

        # execute noise reduction
        with open(INPUT_WAV_PATH, 'wb') as f:
            f.write(audio_data.get_wav_data())
        noisered.create_noisered_wav(INPUT_WAV_PATH, NOISERED_WAV_PATH, BG_WAV_PATH)

        # load or get model
        if threading.get_ident() not in model_map.models:
            print(f"load model. tid:{threading.get_ident()}")
            model_map.models[threading.get_ident()] = load_model()
        model = model_map.models[threading.get_ident()]

        # create input from wav data
        # io_obj = BytesIO(audio_data.get_wav_data())
        # x = create_mfcc_from_io(io_obj)
        x = create_features(NOISERED_WAV_PATH, FEATURE_TYPE)
        # x = create_mfcc_from_file(INPUT_WAV_PATH)

        # complement shortage space
        print(f"x:{x.shape},{x.dtype} framedata:{len(audio_data.frame_data)}")
        if x.shape[0] < Tx:
            # min_val = np.amin(x, axis=0)
            # print(f"min_val:{min_val.shape}")
            # calc remaining space size
            empty_space_size = Tx - x.shape[0]
            # create remaining space
            # empty_space = np.tile(min_val, (empty_space_size, 1))
            empty_space = np.zeros((empty_space_size, n_freq), dtype=np.float32)
            # complement data's empty space
            print(f"emptysp:{empty_space.shape}")
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


def extract_silence(raw_data: bytearray, percentile=75) -> Union[AudioSegment, None]:
    # generate the WAV file contents
    wav_io = BytesIO(raw_data)
    segment = AudioSegment.from_wav(wav_io)
    dbfs_list = [segment[i:i + 1].dBFS for i in range(len(segment))]
    smoothed_dbfs_list = np.convolve(dbfs_list, np.array([1.0/10.0 for _ in range(10)])[::-1], 'same')
    std = np.std(smoothed_dbfs_list)
    if std < 3.5:
        # treat as silence whole time.
        return segment
    threshold = np.percentile(dbfs_list, percentile)

    step_size = 500
    extract_size = 3000
    print(f"segmentsize:{len(segment)} std:{np.std(smoothed_dbfs_list)} threshold:{threshold}")
    for i in np.arange(0, len(segment), step_size):
        if i + extract_size >= len(segment):
            # silent part is not found
            return None
        # print(f"threadhold:{threshold} vals:{smoothed_dbfs_list[i:i+extract_size][:30]}")
        if all([v < threshold for v in smoothed_dbfs_list[i:i+extract_size]]):
            return segment[i:i+extract_size]
    else:
        # silent part is not found
        return None


def listen_background():
    background_listener = noisered.BackgroundListener()
    with Microphone(sample_rate=SAMPLE_RATE) as source:
        background_listener.adjust_for_ambient_noise(source)
        while os.path.exists(SHARED_MEM_DIR):
            audio_data = background_listener.listen(source, pause_time_limit=5)
            if not audio_data:
                time.sleep(1)
                continue

            segment = extract_silence(audio_data.get_wav_data())
            if not segment:
                time.sleep(1)
                continue

            with noisered.SEMAPHORE:
                try:
                    os.remove(BG_WAV_PATH)
                except:
                    pass
                    # create wav file
                segment.export(BG_WAV_PATH, format='wav', bitrate=256)
                print(f"export bgm. {BG_WAV_PATH}. size={len(segment)}")
                # with open(BG_WAV_PATH, 'wb') as f:
                #    f.write(audio_data.get_wav_data())


def start_listen_background():
    t = threading.Thread(target=listen_background, name='listen_background')
    t.start()


def main():
    # start to listen background with another thread.
    start_listen_background()

    while not os.path.exists(BG_WAV_PATH):
        print('ready for bg wav ...')
        time.sleep(1)

    # initialize recognizer
    recognizer = Recognizer()
    recognizer.speaking_duration = 0.1
    recognizer.phrase_threshold = 0.1
    with Microphone(sample_rate=SAMPLE_RATE) as source:
        # listen for 1 second to calibrate the energy threshold for ambient noise levels
        recognizer.adjust_for_ambient_noise(source)
        print("Calibrated. Say something!")

    source = Microphone(sample_rate=SAMPLE_RATE)
    with Pool(THREAD_NUM) as pool:
        callback_with_model = partial(callback, model_map=ModelMap(), pool=pool)
        recognizer.listen_in_background(source, callback_with_model, PHRASE_TIME_LIMIT)
        while True:
            time.sleep(10)
        # pool.terminate()


if __name__ == '__main__':
    try:
        # initialize_shared_mem_dir()
        main()
    finally:
        pass
        # remove_shared_mem_dir()
