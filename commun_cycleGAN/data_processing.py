#@title Data preprocessing

import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

DATASET = "vangogh2photo" #@param ["apple2orange", "summer2winter_yosemite", "horse2zebra", "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "maps", "cityscapes", "facades", "iphone2dslr_flower"] {allow-input: true}
TF_DATASET = 'cycle_gan/' + DATASET

# load dataset
tf_dataset, info = tfds.load(TF_DATASET, batch_size=-1, with_info=True)
np_dataset = tfds.as_numpy(tf_dataset)
train_A, train_B, test_A, test_B = np_dataset["trainA"]["image"] / 127.5 - 1., np_dataset["trainB"]["image"] / 127.5 - 1., np_dataset["testA"]["image"] / 127.5 - 1., np_dataset["testB"]["image"] / 127.5 - 1.
train_A, train_B, test_A, test_B = train_A.astype('float32'), train_B.astype('float32'), test_A.astype('float32'), test_B.astype('float32')

DIMS = train_A[0].shape
DATASET = DATASET.upper()


PATHS = [
         f'plots/{DATASET}/',
         f'models/{DATASET}/',
         f'logs/{DATASET}/'      
          ]

for path in PATHS:
    Path(path).mkdir(parents=True, exist_ok=True)