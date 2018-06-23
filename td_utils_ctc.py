import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment
import pandas as pd
import numpy as np
import pickle
from scipy.signal import spectrogram
from python_speech_features import mfcc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import char_map

# import pydevd
# pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)

# sampling time
AUDIO_DURATION = 10000
# The number of time steps input to the model from the spectrogram
Tx = 999
# Number of frequencies input to the model at each time step of the spectrogram
n_freq = 26
# duration of activation time
ACTIVE_DURATION = 20
# dataset ratio
TRAIN_RATIO = 0.95
VALIDATE_RATIO = 0.05


# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    else:
        raise RuntimeError(f"invalid channels. file={wav_file}")
    return pxx


# Calculate spectrogram for a wav audio file
def create_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        freqs, bins, pxx = spectrogram(data, fs, nperseg=nfft, noverlap=noverlap)
    elif nchannels == 2:
        freqs, bins, pxx = spectrogram(data[:, 0], fs, nperseg=nfft, noverlap=noverlap)
    else:
        raise RuntimeError(f"invalid channels. file={wav_file}")
    return pxx


def plot_mfcc_feature(vis_mfcc_feature):
    # plot the MFCC feature
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(vis_mfcc_feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xticks(np.arange(0, 13, 2), minor=False)
    plt.show()


# Calculate mfcc for a wav audio file
def create_mfcc(wav_file):
    rate, data = get_wav_info(wav_file)
    mfcc_dim = n_freq
    nchannels = data.ndim

    if nchannels == 1:
        features = mfcc(data, rate, numcep=mfcc_dim)
    elif nchannels == 2:
        features = mfcc(data[:, 0], rate, numcep=mfcc_dim)
    else:
        raise RuntimeError(f"invalid channels. file={wav_file}")
    return features


# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# Load raw audio files for speech synthesis
def load_raw_audio():
    audio_dir_path = "./train/audio/"
    keywords = {}
    backgrounds = []
    for keyword in os.listdir(audio_dir_path):
        if keyword.startswith("_"):
            continue
        if keyword not in keywords:
            keywords[keyword] = []
        for filename in os.listdir(audio_dir_path + keyword):
            if not filename.endswith("wav"):
                continue
            audio = AudioSegment.from_wav(audio_dir_path + keyword + "/" + filename)
            keywords[keyword].append({'audio': audio, 'filename': filename})
    for filename in os.listdir(audio_dir_path + "_background_noise_/"):
        if not filename.endswith("wav"):
            continue
        audio = AudioSegment.from_wav(audio_dir_path + "_background_noise_/" + filename)
        backgrounds.append({'audio': audio, 'filename': filename})
    return keywords, backgrounds


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    # Make sure segment doesn't run past the AUDIO_DURATION sec background
    segment_start = np.random.randint(low=0, high=AUDIO_DURATION-segment_ms)
    segment_end = segment_start + segment_ms - 1

    return segment_start, segment_end


def get_random_time_segment_bg(background):
    """
    Gets background file positions

    Arguments:
    background -- background AudioSegment

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    bg_duration = len(background)
    # Make sure segment doesn't run past the 10sec background
    segment_start = np.random.randint(low=0, high=bg_duration - AUDIO_DURATION)
    segment_end = segment_start + AUDIO_DURATION

    return segment_start, segment_end


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    ### START CODE HERE ### (≈ 4 line)
    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if previous_start <= segment_start <= previous_end + ACTIVE_DURATION or \
                previous_start <= segment_end <= previous_end + ACTIVE_DURATION:
            overlap = True
    ### END CODE HERE ###

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)

    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


def create_training_sample(background, keywords, filename='sample.wav', feature='mfcc',
                           is_train=True):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    keywords -- a list of audio segments of each word

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    # Make background quieter
    output = background - 20

    y_words = {}
    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    # number_of_keywords = np.random.randint(0, 5)
    number_of_keywords = np.random.choice([0, 1, 1, 2, 2, 3, 3, 3, 4, 4])
    random_keyword_indices = np.random.randint(len(keywords), size=number_of_keywords)
    random_keywords = [(i, keywords[list(keywords.keys())[i]]) for i in random_keyword_indices]
    # print(f"num:{number_of_keywords}")
    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for idx, random_keyword_list in random_keywords:
        # print(f"idx:{idx}")
        sample_num = len(random_keyword_list)
        train_num = int(sample_num * TRAIN_RATIO)
        if is_train:
            random_index = np.random.randint(train_num)
        else:
            validate_num = int(sample_num * VALIDATE_RATIO)
            random_index = train_num + np.random.randint(validate_num)
        random_keyword = random_keyword_list[random_index]
        audio_data = random_keyword['audio']
        audio_filename = random_keyword['filename']

        # Insert the audio clip on the background
        output, segment_time = insert_audio_clip(output, audio_data, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y" with increment
        keyword = list(keywords.keys())[idx]
        # print(f"keyword:{keyword} start:{segment_start} end:{segment_end} audio_file:{audio_filename}")
        # print(f"create_training_sample1 y:{y[701:]}")
        if keyword in KEYWORDS.keys():
            y_words[segment_start] = keyword

        # print(f"create_training_sample2 y:{y[701:]}")
        # Standardize the volume of the audio clip
        output = match_target_amplitude(output, -20.0)

    # Export new training example
    file_handle = output.export(filename, format="wav")
    # print(f"File ({filename}) was saved in your directory.")

    # Get features of the new recording (background with superposition of positive and negatives)
    if feature == 'mfcc':
        x = create_mfcc(filename)
    else:
        x = create_spectrogram(filename)

    keyword_times = sorted(list(y_words.keys()))
    if len(y_words.keys()) == 0:
        y_sequence = " "
    elif keyword_times[0] < ACTIVE_DURATION // 2:
        y_sequence = ""
    else:
        y_sequence = " "
    for segment in keyword_times:
        y_sequence += y_words[segment] + " "
    y = []
    for char in y_sequence:
        y.append(char_map.CHAR_MAP[char])
    # print(f"y_words:{y_words} y_seq:'{y_sequence}' y:{y} decoded_y:'{int_sequence_to_text(y)}'")
    return x, y


def normalize(features, mean, std, eps=1e-14):
    """ Center a feature using the mean and std
    Params:
        feature (numpy.ndarray): Feature to normalize
    """
    # return (feature - np.mean(feature, axis=0)) / (np.std(feature, axis=0) + eps)
    return (features - mean) / (std + eps)


def normalize_dataset(features_list):
    flatten = np.vstack([features for features in features_list])
    # print(f"originshape:{features_list.shape} flattenshape:{flatten.shape}")
    mean = np.mean(flatten, axis=0)
    std = np.std(flatten, axis=0)
    for i, features in enumerate(features_list):
        features_list[i] = normalize(features, mean, std)


def create_dataset(backgrounds, keywords, is_train=True):
    # extract background
    background = backgrounds[np.random.randint(len(backgrounds))]
    background_audio_data = background['audio']
    bg_segment = get_random_time_segment_bg(background_audio_data)
    clipped_background = background_audio_data[bg_segment[0]:bg_segment[1]]

    feat, label = create_training_sample(clipped_background, keywords, is_train=is_train)
    return feat, label


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map.CHAR_MAP['<SPACE>']
        else:
            ch = char_map.CHAR_MAP[c]
        int_sequence.append(ch)
    return int_sequence


def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        ch = char_map.INDEX_MAP[c]
        text.append(ch)
    return text


KEYWORDS = {
    'silence': 0,
    'unknown': 1,  # others
    'up': 2,
    'down': 3,
    'off': 4,
    'on': 5,
    'yes': 6,
    'no': 7,
    'right': 8,
    'stop': 9,
    'left': 10,
    'go': 11,
    'tree': 12,
    'zero': 13,
    'one': 14,
    'two': 15,
    'three': 16,
    'four': 17,
    'five': 18,
    'six': 19,
    'seven': 20,
    'eight': 21,
    'nine': 22,
    'bed': 23,
    'bird': 24,
    'cat': 25,
    'dog': 26,
    'happy': 27,
    'house': 28,
    'marvin': 29,
    'sheila': 30,
    'wow': 31,
}

# KEYWORDS (exclude silence)
NUM_CLASSES = len(KEYWORDS)
# one-hot-encoding (exclude silence)
KEYWORDS_INDEX_DF = pd.get_dummies(list(range(NUM_CLASSES)))


def get_keyword_index(keyword):
    if keyword in KEYWORDS:
        # target word
        # decrement index offset
        return KEYWORDS[keyword]
    else:
        # unknown
        return KEYWORDS['unknown']
