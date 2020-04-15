#Build and compile the model with its optimizers

from tensorflow.keras.optimizers import Adam
from cycleGAN import *
from gan_networks import *
from buffer import *


def build_cycleGAN(ALPHA, BETA_1, DIMS, DATASET, BUFFER_MAX_SIZE) :

    #in : params of the model
    #out : complete Model object of cycleGAN

    disc_A, disc_B, gen_A_to_B, gen_B_to_A = get_networks (DIMS, DATASET)

    buffer_A = Buffer(max_size = BUFFER_MAX_SIZE)
    buffer_B = Buffer(max_size = BUFFER_MAX_SIZE)

    # optimizers
    gen_A_to_B_optimizer = Adam(lr=ALPHA, beta_1=BETA_1)
    gen_B_to_A_optimizer = Adam(lr=ALPHA, beta_1=BETA_1)
    disc_A_optimizer = Adam(lr=ALPHA, beta_1=BETA_1)
    disc_B_optimizer = Adam(lr=ALPHA, beta_1=BETA_1)

    # model
    model = cycleGAN(name=f'cycleGAN_{DATASET}',
                    disc_A = disc_A,
                    disc_B = disc_B,
                    gen_A_to_B = gen_A_to_B,
                    gen_B_to_A = gen_B_to_A,
                    disc_A_optimizer = disc_A_optimizer,
                    disc_B_optimizer = disc_B_optimizer,
                    gen_A_to_B_optimizer = gen_A_to_B_optimizer,
                    gen_B_to_A_optimizer = gen_B_to_A_optimizer,
                    buffer_A = buffer_A,
                    buffer_B = buffer_B,
                                            )


    return model