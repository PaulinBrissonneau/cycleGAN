#@title Data preprocessing

import tensorflow as tf
import tensorflow_datasets as tfds
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


    return train_A, train_B, test_A, test_B, DIMS, DATASET


#c'est du mapping, donc moins de mémoire vive
#il faut avoir les images enregistrées sur le disque (mais si qqun se chauffe on peut le faire depuis tfds aussi)
def get_datas_mapping(test_ratio, data_x_folder, data_y_folder, debug_sample):

    #in : dataset (str)
    #out : train_A, train_B, test_A, test_B

    files_x = [data_x_folder + img for img in os.listdir(data_x_folder)]
    files_y = [data_y_folder + img for img in os.listdir(data_y_folder)]

    #split in test and train OR debug_sample (smaller sample to see bugs)
    if debug_sample == -1 : nb_train = int(len(files_x)*(1-test_ratio))
    else : nb_train = debug_sample

    files_x_train, files_y_train = files_x[:nb_train], files_y[:nb_train]
    files_x_test, files_y_test = files_x[nb_train:], files_y[nb_train:]

    def make_dataset (files) :

        nb = len(files)
        print("files imported :",nb)
        filenames = tf.constant(files)
        dataset = tf.data.Dataset.from_tensor_slices((filenames))
       
        #mapping dataset <-> images
        def _parse_function(filenames):
            image_string = tf.io.read_file(filenames)  
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            #image = (image-(255/2))/255
            image = image / 127.5 - 1.
            return image
            
        dataset = dataset.map(_parse_function)
        return dataset, nb

    ## TOUT LE RESTE EST UN AJOUT POUR LA COMPATIBLITE AVEC ILYAS, MAIS A RECODER PROPREMENT

    #get dimnensions of an image
    image_string = tf.io.read_file(files_x_train[0])  
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    dims = image.shape

    #create mapped datasets
    train_A, nb_train_A = make_dataset (files_x_train)
    train_B, nb_train_B = make_dataset (files_y_train)
    test_A, nb_test_A = make_dataset (files_x_test)
    test_B, nb_test_B = make_dataset (files_y_test)

    #on peut ajouter des shuffles ici si vous voulez

    return train_A, train_B, test_A, test_B, dims