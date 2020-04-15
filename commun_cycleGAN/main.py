# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import tensorflow as tf

from config_reader import *
from data_process import *
from display import *
from saver import *
from cycleGAN_builder import *

#passage en eager execution
tf.config.experimental_run_functions_eagerly(True)

#read configuration file
CONFIG = read_config("config.json")

#passage en GPU
if CONFIG['on_gpu'] :
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)


"""trucs à faire ici :
- faire le config_reader - OK 
- gérer l'affichage
- gérer les datas test
- refair le buffer
- ...
"""



train_A, train_B, test_A, test_B, DIMS = get_datas_mapping(test_ratio = CONFIG['test_ratio'])


# a pandas dataframe to save the loss information to
losses = DataFrame(columns = ['A_to_B_loss', 'B_to_A_loss'])
losses.loc[len(losses)] = (0, 0)

#pour l'instant incompatible
#plot_sample(VIS_LINES, VIS_ROWS, PLOT_SIZE)

model = build_cycleGAN(CONFIG['alpha'], CONFIG['beta_1'], DIMS, CONFIG['dataset'], CONFIG['max_buffer_size'])


#pour l'instant incompatible
#save_performance(START_EPOCH-1, model, train_A, train_B, losses, DATASET)

BATCH_SIZE = CONFIG['batch_size']
train_A, train_B, test_A, test_B = train_A.batch(BATCH_SIZE), train_B.batch(BATCH_SIZE), test_A.batch(BATCH_SIZE), test_B.batch(BATCH_SIZE)


#continue training
if not CONFIG['load_model'] :
    START_EPOCH = 0
else:
    START_EPOCH = CONFIG['load_epoch']
    

# iterate through epochs
for i in range(START_EPOCH, CONFIG['end_epoch']):

    # initiate loss counter
    loss = []

    #get the bumber of batch
    number_of_batch = len(list(zip(train_A, train_B)))

    tqdm_bar = tqdm(total=number_of_batch)

    #enumerate batches over the training set (Joanna the Best)
    for real_a, real_b in zip(train_A, train_B) :
        tqdm_bar.update(1)

        # update generator B->A via adversarial and cycle loss
        B_to_A_loss = model.train_B_to_A(real_a, real_b)
        # update discriminator for A -> [real/fake]
        disc_A_loss = model.train_disc_A(real_a, real_b)
        # update generator A->B via adversarial and cycle loss
        A_to_B_loss = model.train_A_to_B(real_a, real_b)
        # update discriminator for B -> [real/fake]
        disc_B_loss = model.train_disc_B(real_a, real_b)
        # save batch loss
        loss.append((A_to_B_loss, B_to_A_loss))
    # average loss over epoch
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    
    # evaluate the model performance, sometimes

    #pour l'instant incompatible
    #save_performance(i, model, train_A, train_B, losses, DATASET)
    