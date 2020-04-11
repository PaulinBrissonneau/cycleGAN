#@title Data preprocessing

import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import os

def get_datas_ilyas (dataset):
    #in : dataset (str)
    #out : train_A, train_B, test_A, test_B

    DATASET = dataset #@param ["apple2orange", "summer2winter_yosemite", "horse2zebra", "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "maps", "cityscapes", "facades", "iphone2dslr_flower"] {allow-input: true}
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

    return train_A, train_B, test_A, test_B, DIMS, DATASET


#il faut avoir les images enregistrées sur le disque
#c'est du mapping, donc moins de mémoire vive
def get_datas_paulin(dataset, test_ratio):

    #à coder : compatibilité avec l'autre code : 

    #in : dataset (str)
    #out : train_A, train_B, test_A, test_B

    direct_x = "/home/paulin/Documents/datas/orangeApple/orange/"
    direct_y = "/home/paulin/Documents/datas/orangeApple/apple/"

    files_x = [direct_x + img for img in os.listdir(direct_x)]
    files_y = [direct_y + img for img in os.listdir(direct_y)]

    nb_train = int(len(files_x)*(1-test_ratio))

    files_x_train = files_x[:nb_train]
    files_y_train = files_y[:nb_train]
    files_x_test = files_x[nb_train:]
    files_y_test = files_y[nb_train:]

    def make_dataset (files) :

        nb = len(files)
        print("len(files_x_train) :",nb)
        filenames = tf.constant(files)
        dataset = tf.data.Dataset.from_tensor_slices((filenames))
       
        def _parse_function(filenames):
            image_string = tf.io.read_file(filenames)  
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            image = (image-(255/2))/255
            return image
            
        dataset = dataset.map(_parse_function)
        return dataset

    ## AJOUT COMPATIBLITE AVEC ILYAS, MAIS A RECODER
    image_string = tf.io.read_file(files_x_train[0])  
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    dims = image.shape

    print(len(files_x_train))
    print(len(files_x_test))

    return make_dataset (files_x_train), make_dataset (files_y_train), make_dataset (files_x_test), make_dataset (files_y_test), dims, dataset