import mpl_toolkits  # import before pathlib
import sys
from pathlib import Path

from tensorflow import set_random_seed

sys.path.append(Path(__file__).parent)
from train_model import *
from single_word_model import *
from dataset import *

np.random.seed(19)
set_random_seed(19)

weight_param_path = f"model/kmn_kaggle_cnn_1sec_{FEATURE_TYPE}.weights.best.hdf5"

test_model = create_model_cnn(input_shape=(Tx, n_freq), is_train=True)
test_model.load_weights(weight_param_path)

# stop
test_wave = 'test/audio/clip_0a0c03051.wav'

def predict(test_wave, test_model):
    x = create_features(test_wave, FEATURE_TYPE)
    # padding shortage space
    if x.shape[0] < Tx:
        # calc remaining space size
        empty_space_size = Tx - x.shape[0]
        # create remaining space
        empty_space = np.zeros((empty_space_size, n_freq), dtype=np.float32)
        # complement data's empty space
        x = np.concatenate((x, empty_space), axis=0)
    x = np.float32(np.array([x]))

    # predict
    predicted = test_model.predict(x)
    pred_y = np.argmax(predicted[0], axis=0)
    predicted_word = command.REVERSE_KEYWORDS_MAP[pred_y]
    return pred_y, predicted_word

OUTPUT_FILE = 'test_dataset_prediction.txt'
DIR = './test/audio/'

content = "fname,label\n"
for test_file in os.listdir(DIR):
    test_path = os.path.join(DIR, test_file)
    if not os.path.isfile(test_path):
        continue
    pred_y, predicted_word = predict(test_path, test_model)
    content += f"{test_file},{predicted_word}\n"

with open(OUTPUT_FILE, 'w') as f:
    f.write(content)

