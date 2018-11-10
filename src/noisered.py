import threading
import os
import traceback
import collections
import math
import audioop

import sox
from speech_recognition import Recognizer
from speech_recognition import AudioData
from speech_recognition import AudioSource
from speech_recognition import Microphone
from speech_recognition import WaitTimeoutError
import pyaudacity

SEMAPHORE = threading.Semaphore(1)


def create_noisered_wav(in_wav, out_wav, bg_wav):
    """

    :param in_wav:
    :param out_wav:
    :param bg_wav:
    :return:

    >>> create_noisered_wav('/var/tmp/keyword_recognizer/input.wav', '/var/tmp/keyword_recognizer/noisered.wav', '/var/tmp/keyword_recognizer/bg_input.wav')
    execute noise reduction. input=/var/tmp/keyword_recognizer/input.wav output=/var/tmp/keyword_recognizer/noisered.wav bg_wav=/var/tmp/keyword_recognizer/bg_input.wav

    1: »0⋅1/1⋅1/1⋅1/1
    """
    with SEMAPHORE:
        result = pyaudacity.noisered(bg_wav, 0, 0.5, in_wav, 12, 6, 3, out_wav + '.tmp1')
        if not os.path.exists(out_wav + '.tmp1'):
            print(f"{out_wav + '.tmp1'} does not exists")
            return False
        if not result:
            return False
        result = pyaudacity.noisered(out_wav + '.tmp1', 0, 0.25, out_wav + '.tmp1', 12, 6, 3, out_wav + '.tmp2')
        if not os.path.exists(out_wav + '.tmp2'):
            print(f"{out_wav + '.tmp2'} does not exists")
            return False
        if result:
            try:
                os.remove(out_wav)
            except:
                pass
            os.rename(out_wav + ".tmp2", out_wav)
            print(f"execute noise reduction. input={in_wav} output={out_wav} bg_wav={bg_wav}")
            return True


class BackgroundListener(Recognizer):
    def __enter__(self):
        raise NotImplementedError("unimplemented")

    def __exit__(self, exc_type, exc_value, trace_back):
        raise NotImplementedError("unimplemented")

    def __init__(self):
        super().__init__()
        self.pause_threshold = 3

    def listen(self, source: Microphone, timeout=None, pause_time_limit=None, _=None):
        """
        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance, which it returns.

        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold`` (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold`` seconds of non-speaking or there is no more audio input. The ending silence is not included.

        The ``timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``, there will be no wait timeout.

        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, there will be no phrase time limit.

        This operation will always complete within ``timeout + phrase_timeout`` seconds if both are numbers, either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
        """
        assert isinstance(source, AudioSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before listening, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))

        # read audio input for pause until there is a pause that is long enough
        # number of seconds of audio read
        elapsed_time = 0
        frames = collections.deque()

        # store audio input until the pause starts
        while True:
            # handle waiting too long for pause by raising an exception
            elapsed_time += seconds_per_buffer
            if timeout and elapsed_time > timeout:
                raise WaitTimeoutError("listening timed out while waiting for pause to start")

            buffer = source.stream.read(source.CHUNK)
            # an empty buffer means that the stream has ended and there is no data left to read
            if len(buffer) == 0:
                # reached end of the stream
                break
            # detect whether non speaking has started on audio input
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal
            if energy < self.energy_threshold:
                frames.append(buffer)
                break

            # dynamically adjust the energy threshold using asymmetric weighted average
            if self.dynamic_energy_threshold:
                # account for different chunk sizes and rates
                damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
                target_energy = energy * self.dynamic_energy_ratio
                self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

        # read audio input until the pause ends
        phase_count, pause_count = 0, 0
        pause_start_time = elapsed_time
        while True:
            # handle pause being too long by cutting off the audio
            elapsed_time += seconds_per_buffer
            if pause_time_limit and elapsed_time - pause_start_time > pause_time_limit:
                break

            buffer = source.stream.read(source.CHUNK)
            if len(buffer) == 0:
                # reached end of the stream
                break

            # check if speaking has stopped for longer than the pause threshold on the audio input
            # unit energy of the audio signal within the buffer
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)
            if energy < self.energy_threshold:
                frames.append(buffer)
                pause_count += 1
            else:
                # detect sound
                break
        if pause_count > pause_buffer_count:
            frame_data = b''.join(frames)
            return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
        else:
            # buffer size could not reach threshold.
            return None
