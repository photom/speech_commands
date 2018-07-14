import threading
import os
import traceback

import sox


SEMAPHORE = threading.Semaphore(1)


def create_noiseprof(in_wav, profile_path, trim_start=0, trim_end=0.5):
    try:
        tmp_profile_path = f"{profile_path}.tmp"

        # create transformer
        tfm = sox.Transformer()
        # trim the audio between start and end seconds.
        tfm.trim(trim_start, trim_end)
        tfm.noiseprof(in_wav, tmp_profile_path)
        # create the output file.
        tfm.build(in_wav, None)

        # move output
        with SEMAPHORE:
            try:
                os.remove(profile_path)
            except:
                pass
            os.rename(tmp_profile_path, profile_path)
    except:
        traceback.print_exc()
        raise

    # see the applied effects
    # print(f"created profile. input={in_wav} profile={profile_path} {tfm.effects_log}")


def create_noisered_wav(in_wav, out_wav, profile_path, amount=0.2):
    with SEMAPHORE:
        # create transformer
        tfm = sox.Transformer()
        # trim the audio between start and end seconds.
        tfm.noisered(profile_path, amount=amount)
        # create the output file.
        tfm.build(in_wav, out_wav)
        # see the applied effects
        print(f"execute noise reduction. input={in_wav} output={out_wav} profile={profile_path} {tfm.effects_log}")
