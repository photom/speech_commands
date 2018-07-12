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

# import pydevd
# pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)

# sampling time
AUDIO_DURATION = 2000
# The number of time steps input to the model from the spectrogram
Tx = 198
# Number of frequencies input to the model at each time step of the spectrogram
n_freq = 26
# The number of time steps in the output of our model
Ty = 188
# interval
KEYWORD_INTERVAL_MIN = 50
KEYWORD_INTERVAL_MAX = 500
ACTIVE_AUDIO_OFFSET = 100
# dataset ratio
TRAIN_RATIO = 0.95
VALIDATE_RATIO = 0.05
WINDOW_SIZE_SAMPLE = 0.03
WINDOW_STRIDE_SAMPLES = 0.01
# How much of the training data should be known words.
KNOWN_KEYWORD_RATIO = 0.75


class RawData:
    def __init__(self, keywords: dict, backgrounds: np.ndarray,
                 mean: np.ndarray, std: np.ndarray):
        self.keywords = keywords
        self.backgrounds = backgrounds
        self.mean = mean
        self.std = std

    @staticmethod
    def get_known_keywords():
        return list(KNOWN_KEYWORDS.keys())

    def get_unknown_keywords(self):
        return list(set(self.keywords.keys()) - set(KNOWN_KEYWORDS.keys()))


class KeywordAudioData:
    def __init__(self, audio: AudioSegment, keyword: str, filename: str):
        self.audio = audio
        self.keyword = keyword
        self.filename = filename
        self.value = self.get_keyword_index()
        self.start_ms, self.end_ms = self.get_active_range()

    def get_active_range(self, percentile1=75, percentile2=50):
        dbfs_list = [self.audio[i:i+1].dBFS for i in range(len(self.audio))]
        threshold = np.percentile(dbfs_list, percentile1)
        if not any([dbfs >= threshold for dbfs in dbfs_list]):
            threshold = np.percentile(dbfs_list, percentile2)
            print(f"missing peek. file:{self.filename}")

        values = []
        for i in range(len(self.audio)):
            if self.audio[i:i+1].dBFS >= threshold:
                values.append(self.value)
            else:
                values.append(SILENCE_KEYWORD_IDX)
        # print(self.keyword)
        # for i in range(len(self.audio)):
        #    print(f"{i}: {values[i]}")
        # mfcc_val = create_mfcc(self.filename)
        # print(mfcc_val)
        # for i in range(len(mfcc_val)):
        #    print(f"{i}: {mfcc_val[i]}")
        try:
            first_idx = values.index(self.value)
            last_idx = len(values) - values[::-1].index(self.value) - 1
        except Exception as e:
            print(f"error:{e} file:{self.filename}")
            raise
        return first_idx, last_idx

    def get_keyword_index(self):
        if self.keyword in KNOWN_KEYWORDS:
            # target word
            # decrement index offset
            return KNOWN_KEYWORDS[self.keyword]
        else:
            # unknown
            return UNKNOWN_KEYWORD_IDX


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
    samplerate, data = get_wav_info(wav_file)
    mfcc_dim = n_freq
    nchannels = data.ndim

    if nchannels == 1:
        features = mfcc(data, samplerate=samplerate, numcep=mfcc_dim,
                        winlen=WINDOW_SIZE_SAMPLE, winstep=WINDOW_STRIDE_SAMPLES)
    elif nchannels == 2:
        features = mfcc(data[:, 0], samplerate=samplerate, numcep=mfcc_dim,
                        winlen=WINDOW_SIZE_SAMPLE, winstep=WINDOW_STRIDE_SAMPLES)
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


def create_features(filename, feature_type):
    if feature_type == 'mfcc':
        x = create_mfcc(filename)
    else:
        x = create_spectrogram(filename)
    return x


def get_statistics(features_list):
    features_list = np.vstack(features_list)
    mean = np.mean(features_list, axis=0)
    std = np.std(features_list, axis=0)
    print(f"get_statistics:{features_list.shape} mean:{mean.shape} std:{std.shape}")
    return mean, std


# Load raw audio files for speech synthesis
def load_raw_audio(feature_type='mfcc'):
    audio_dir_path = "./train/audio/"
    keywords = {}
    backgrounds = []
    features_list = []
    # keyword
    for keyword in os.listdir(audio_dir_path):
        if keyword.startswith("_"):
            continue
        if keyword not in keywords:
            keywords[keyword] = []
        for filename in os.listdir(audio_dir_path + keyword):
            if not filename.endswith("wav"):
                continue
            file_path = os.path.join(audio_dir_path, keyword, filename)
            audio = AudioSegment.from_wav(file_path)
            keywords[keyword].append(KeywordAudioData(audio, keyword, filename))
            # get features
            features_list.append(create_features(file_path, feature_type))
    # background
    for filename in os.listdir(os.path.join(audio_dir_path, "_background_noise_")):
        if not filename.endswith("wav"):
            continue
        file_path = os.path.join(audio_dir_path, "_background_noise_", filename)
        audio = AudioSegment.from_wav(file_path)
        backgrounds.append({'audio': audio, 'filename': filename})
        # get features
        # workaround for WavFileWarning: Chunk (non-data)
        for i in range(len(audio) // 1000):
            background_clipped = audio[i * 1000:i * 1000 + 1000]
            background_clipped.export('sample.wav', format="wav")
            features_list.append(create_features('sample.wav', feature_type))

    # calc statistics
    mean, std = get_statistics(np.array(features_list))

    # shuffle audio list
    backgrounds = np.array(backgrounds)
    np.random.shuffle(backgrounds)
    for keyword in keywords.keys():
        keywords[keyword] = np.array(keywords[keyword])
        np.random.shuffle(keywords[keyword])
    return RawData(keywords, backgrounds, mean, std)


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    # Make sure segment doesn't run past the AUDIO_DURATION sec background
    segment_start = np.random.randint(low=0, high=AUDIO_DURATION - segment_ms)
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


def calc_slide_size(segment_time, previous_segment):
    """
    calculate slide width if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segment -- tuple of (segment_start, segment_end) for the existing segment

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    if not previous_segment:
        return 0

    segment_start, segment_end = segment_time
    slide_size = np.random.choice(list(range(KEYWORD_INTERVAL_MIN, KEYWORD_INTERVAL_MAX)))
    # print(f"slidesize:{slide_size}")
    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    previous_start, previous_end = previous_segment
    if previous_start - ACTIVE_AUDIO_OFFSET <= segment_start <= previous_end + ACTIVE_AUDIO_OFFSET:
        # overlaid, add offset
        return previous_end + slide_size - segment_start
    elif previous_start - ACTIVE_AUDIO_OFFSET<= segment_end <= previous_end + ACTIVE_AUDIO_OFFSET:
        # overlaid, subtract offset
        return segment_end - previous_start - slide_size
    else:
        # not overlay
        return 0


def insert_ones(y: np.ndarray, segment_start_ms: int, audio: KeywordAudioData):
    """
    Update the label vector y during segment.

    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_start_ms -- the start time of the segment in ms
    audio -- target keyword audio
    truncated -- whether the keyword is truncated

    Returns:
    y -- updated labels
    """
    # default keyword index
    value = audio.value

    segment_start_y = int((segment_start_ms + audio.start_ms) * Ty / float(AUDIO_DURATION))
    if segment_start_y < 0:
        # truncate exceeding range
        segment_start_y = 0
        # consider truncated word as invalid
        value = UNKNOWN_KEYWORD_IDX
        # print("set unknown idx")
    segment_end_y = int((segment_start_ms + audio.end_ms) * Ty / float(AUDIO_DURATION))
    if segment_end_y > Ty:
        # truncate exceeding range
        segment_end_y = Ty
        # consider truncated word as invalid
        value = UNKNOWN_KEYWORD_IDX
        # print("set unknown idx")

    # print(f"keyword:{audio.keyword} file:{audio.filename} start:{audio.start_ms} end:{audio.end_ms} segstart:{segment_start_ms} segend:{segment_end_ms} segstart_y:{segment_start_y} segend_y:{segment_end_y}")
    # print(f"insert_ones segment_end_ms:{segment_end_ms} Ty:{Ty} segment_end_y:{segment_end_y} value:{value}")
    # Add one-hot encoded value to the correct index in the background label (y)
    for i in range(segment_start_y, segment_end_y):
        y[i] = value
    # print(f"insert_ones y.shape:{y.shape} y:{y[701:]}")
    return y


def insert_audio_clip(tmp_audio: AudioSegment, y: np.ndarray,
                      keyword_audio: KeywordAudioData, previous_segment: tuple):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    tmp_audio -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segment -- time where audio segment haa already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    keyword_duration = len(keyword_audio.audio)
    background_duration = len(tmp_audio)

    # Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(keyword_duration)
    segment_start_ms, segment_end_ms = segment_time
    # print(f"duration:{keyword_duration} segment:{segment_time}")

    # Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    slide_size = calc_slide_size(segment_time, previous_segment)
    segment_time = (segment_start_ms + slide_size, segment_end_ms + slide_size)
    segment_start_ms, segment_end_ms = segment_time
    # print(f"slide_size:{slide_size} segment:{segment_time}")
    if segment_start_ms >= background_duration or segment_end_ms <= 0:
        # invalid range. skip insert.
        return tmp_audio, y, segment_time

    # set y value
    y = insert_ones(y, segment_time[0], keyword_audio)

    # Superpose audio segment and background
    overlaid_audio = keyword_audio.audio
    if segment_start_ms < 0:
        # truncate exceeded part
        overlaid_audio = overlaid_audio[-segment_start_ms:]
        segment_start_ms = 0
        segment_time = 0, segment_end_ms
    if segment_end_ms > background_duration:
        # truncate exceeded part
        # print(f"end:{segment_end_ms - background_duration}")
        end_of_keyword = keyword_duration - (segment_end_ms - background_duration)
        overlaid_audio = overlaid_audio[:end_of_keyword]
        segment_time = segment_start_ms, background_duration
    # overlaid_audio.export(f"overlaied_{keyword_audio.keyword}.wav", format="wav")
    tmp_audio = tmp_audio.overlay(overlaid_audio, position=segment_start_ms)
    return tmp_audio, y, segment_time


def create_training_sample(background: AudioSegment, raw_data: RawData,
                           feature_type: str = 'mfcc',
                           is_train: bool = True,
                           filename: str = 'sample.wav',
                           remove_tmpfile = True):
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
    tmp_audio = background - 40
    # Initialize y (label vector) of zeros
    y = np.zeros((Ty, 1))
    # Initialize segment times as empty list
    previous_segment = None

    # Select 0-2 random "activate" audio clips from the entire list of "activates" recordings
    # number_of_keywords = np.random.randint(0, 5)
    # number_of_keywords = np.random.choice([0, 1, 1, 2, 2, 3, 3, 3, 4, 4])
    # 0: 10%, 1: 50%, 2: 40%
    number_of_keywords = np.random.choice([0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    random_keywords = []
    for i in range(number_of_keywords):
        if np.random.rand() < KNOWN_KEYWORD_RATIO:
            idx = np.random.randint(len(RawData.get_known_keywords()))
            keyword = RawData.get_known_keywords()[idx]
        else:
            idx = np.random.randint(len(raw_data.get_unknown_keywords()))
            keyword = raw_data.get_unknown_keywords()[idx]
        random_keywords.append((keyword, raw_data.keywords[keyword]))

    # print(f"num:{number_of_keywords}")
    # Loop over randomly selected "activate" clips and insert in background
    for keyword, random_keyword_list in random_keywords:
        # print(f"idx:{idx}")
        sample_num = len(random_keyword_list)
        train_num = int(sample_num * TRAIN_RATIO)
        if is_train:
            random_index = np.random.randint(train_num)
        else:
            validate_num = int(sample_num * VALIDATE_RATIO)
            random_index = train_num + np.random.randint(validate_num)
        keyword_audio = random_keyword_list[random_index]

        # Insert the audio clip on the background
        tmp_audio, y, previous_segment = insert_audio_clip(tmp_audio, y, keyword_audio, previous_segment)
        # print(f"keyword:{keyword} range:{previous_segment} audio_file:{keyword_audio.filename} audio_range:{(keyword_audio.start_ms, keyword_audio.end_ms)}")
        # print(f"create_training_sample1 y:{y[701:]}")
        # Standardize the volume of the audio clip
        tmp_audio = match_target_amplitude(tmp_audio, -20.0)

    # Export new training example
    file_handle = tmp_audio.export(filename, format="wav")
    os.fsync(file_handle.fileno())
    # print(f"File ({filename}) was saved in your directory.")

    # Get features of the new recording (background with superposition of positive and negatives)
    x = create_features(filename, feature_type)
    x = normalize(np.array(x), raw_data.mean, raw_data.std)
    if remove_tmpfile:
        os.remove(filename)
    return x, y


def normalize(features, mean: np.ndarray, std: np.ndarray, eps=1e-14) -> float:
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


def create_dataset(raw: RawData, num: int, is_train=True):
    train_dataset_x = []
    train_dataset_y = []
    for i in range(num):
        # extract background
        bg_idx = np.random.randint(len(raw.backgrounds))
        background = raw.backgrounds[bg_idx]
        background_audio_data = background['audio']
        bg_segment = get_random_time_segment_bg(background_audio_data)
        clipped_background = background_audio_data[bg_segment[0]:bg_segment[1]]
        x, y = create_training_sample(clipped_background, raw, is_train=is_train, remove_tmpfile=False)
        train_dataset_x.append(x)
        train_dataset_y.append(y)
        # print(f"create_dataset y:{y[700:]}")
        # print(f"create_dataset traindatasety:{train_dataset_y[0]}")
    train_dataset_x = np.array(train_dataset_x)
    train_dataset_y = np.array(train_dataset_y)
    return train_dataset_x, train_dataset_y


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


KNOWN_KEYWORDS = {
#    'silence': 0,
#    'unknown': 1,  # others
    'up': 2,
    'down': 3,
    'off': 4,
    'on': 5,
    'yes': 6,
    'no': 7,
    'sheila': 8,
#    'go': 8,
#    'stop': 9,
#    'right': 8,
#    'left': 10,
#    'tree': 12,
#    'zero': 13,
#    'one': 14,
#    'two': 15,
#    'three': 16,
#    'four': 17,
#    'five': 18,
#    'six': 19,
#    'seven': 20,
#    'eight': 21,
#    'nine': 22,
#    'bed': 23,
#    'bird': 24,
#    'cat': 25,
#    'dog': 26,
#    'happy': 27,
#    'house': 28,
#    'marvin': 29,
#    'wow': 31,
}

# KEYWORDS (exclude silence)
NUM_CLASSES = len(KNOWN_KEYWORDS) + 2
# one-hot-encoding (exclude silence)
KEYWORDS_INDEX_DF = pd.get_dummies(list(range(NUM_CLASSES)))
SILENCE_KEYWORD_IDX = 0
UNKNOWN_KEYWORD_IDX = 1


def get_keyword_index(keyword):
    if keyword in KNOWN_KEYWORDS:
        # target word
        # decrement index offset
        return KNOWN_KEYWORDS[keyword]
    else:
        # unknown
        return UNKNOWN_KEYWORD_IDX
