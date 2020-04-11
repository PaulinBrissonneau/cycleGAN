#@title Defining the Generator and the Discrimenator sub-models

from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose,     \
                                    Activation, Input, Dense,                 \
                                    Reshape, Flatten, Dropout, LeakyReLU

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf

LOAD_MODEL = False #@param {type:"boolean"}
LOAD_EPOCH = 0 #@param {type:"number"}
N_RESNET = 9 #@param {type:"number"}


# weight initialization
init = RandomNormal(stddev=0.02)

# define the discriminator model
def define_discriminator(image_shape):
    # source image input
    input_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    # define model
    model = Model(input_image, patch_out)
    return model

# generator a resnet block
def resnet_block(input_layer, n_filters):
    # first layer convolutional layer
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g

# define the standalone generator model
def define_generator(image_shape, n_resnet=N_RESNET):
    # image input
    input_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(input_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    for i in range(n_resnet):
        g = resnet_block(g, 256)
    # u128
    g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    output_image = Activation('tanh')(g)
    # define model
    model = Model(input_image, output_image)
    return model


def get_networks (DIMS, DATASET) :

    #écrire les spécs

    if LOAD_MODEL:
        # generators
        # generator: A -> B
        gen_A_to_B = tf.keras.models.load_model(f'models/{DATASET}/cycleGAN_e{LOAD_EPOCH:03}_gen_A_to_B')
        # generator: B -> A
        gen_B_to_A = tf.keras.models.load_model(f'models/{DATASET}/cycleGAN_e{LOAD_EPOCH:03}_gen_B_to_A')

        # discriminators
        # discriminator: A -> [real/fake]
        disc_A = tf.keras.models.load_model(f'models/{DATASET}/cycleGAN_e{LOAD_EPOCH:03}_disc_A')
        # discriminator: B -> [real/fake]
        disc_B = tf.keras.models.load_model(f'models/{DATASET}/cycleGAN_e{LOAD_EPOCH:03}_disc_B')

    else:
        # generators
        # generator: A -> B
        gen_A_to_B = define_generator(DIMS)
        # generator: B -> A
        gen_B_to_A = define_generator(DIMS)

        # discriminators
        # discriminator: A -> [real/fake]
        disc_A = define_discriminator(DIMS)
        # discriminator: B -> [real/fake]
        disc_B = define_discriminator(DIMS)

    return disc_A, disc_B, gen_A_to_B, gen_B_to_A 