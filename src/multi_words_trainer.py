#!/usr/bin/env python

import mpl_toolkits # import before pathlib
import sys
import pathlib

from tensorflow import set_random_seed

# sys.path.append(pathlib.Path(__file__).parent)
from train_model import *
from multi_words_model import *
from dataset import *

np.random.seed(19)
set_random_seed(19)

raw_data_path = f"raw_data_with_{FEATURE_TYPE}.pickle"

# Load audio segments using pydub
# raw_data = load_raw_audio(FEATURE_TYPE)
# with open(raw_data_path, 'wb') as f:
#     pickle.dump(raw_data, f)
# exit(0)

with open(raw_data_path, 'rb') as f:
    raw_data = pickle.load(f)

print("background len: " + str(len(raw_data.backgrounds[0])))
print("number of keyword: " + str(len(raw_data.keywords.keys())))
print("keywords[0] len: " + str(len(raw_data.keywords['on'][0].audio)))
print("keywords[1] len: " + str(len(raw_data.keywords['off'][0].audio)))
# print(f"mean:{raw_data.mean.shape} std:{raw_data.std.shape}")

weight_param_path = f"model/kmn_cnnbidirectctc_{FEATURE_TYPE}.weights.best.hdf5"
# model_dilation
model = build_model(weight_param_path, create_model=create_model_cnn_bidirect)
model.summary()
# model_dilation
np.random.seed(19)
set_random_seed(19)

for i in range(0, 4):
    print(f"num:{i}")
    train_model(model, raw_data, weight_param_path,
                detect_wakeword=False,
                feature_type=FEATURE_TYPE)
