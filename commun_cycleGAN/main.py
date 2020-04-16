# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import tensorflow as tf
from pathlib import Path

from config_reader import *
from data_process import *
from display import *
from saver import *
from cycleGAN_builder import *
import sys


#passage en eager execution
tf.config.experimental_run_functions_eagerly(True)

#read configuration file
config_file = sys.argv[0]
#config_file = "config_example_cpu.json"

print("config_file : ", config_file)

CONFIG = read_config(config_file)

#passage en GPU
if CONFIG['on_gpu'] :
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)


"""trucs à faire ici :
- faire le config_reader - OK 
- gérer l'affichage - OK
- gérer l'enregistrement - OK
- gérer les datas test
- coder le buffer - OK
- ...
"""


train_A, train_B, test_A, test_B, DIMS = get_datas_mapping(test_ratio = CONFIG['test_ratio'], data_x_folder = CONFIG['data_x_folder'], data_y_folder = CONFIG['data_y_folder'])

PATHS = [
            f"{CONFIG['output_folder']}/plots/{CONFIG['dataset']}/",
            f"{CONFIG['output_folder']}/models/{CONFIG['dataset']}/",
            f"{CONFIG['output_folder']}/logs/{CONFIG['dataset']}/" 
        ]

for path in PATHS:
    Path(path).mkdir(parents=True, exist_ok=True)


# a pandas dataframe to save the loss information to
losses = DataFrame(columns = ['A_to_B_loss', 'B_to_A_loss'])
losses.loc[len(losses)] = (0, 0)

# display sample / visualisation
if CONFIG['plot_sample']: plot_sample(train_A, train_B, CONFIG['vis_lines'], CONFIG['vis_rows'], CONFIG['plot_size'])

model = build_cycleGAN(CONFIG['alpha'], CONFIG['beta_1'], DIMS, CONFIG['dataset'], CONFIG['max_buffer_size'])



#make batches from the tf.Dataset
BATCH_SIZE = CONFIG['batch_size']
train_A, train_B, test_A, test_B = train_A.batch(BATCH_SIZE), train_B.batch(BATCH_SIZE), test_A.batch(BATCH_SIZE), test_B.batch(BATCH_SIZE)



#continue training
if not CONFIG['load_model'] :
    START_EPOCH = 0
else:
    START_EPOCH = CONFIG['load_epoch']

# evaluate the model performance, and save
#j'ai laissé le train_A dans le plot comme c'était dans le code, mais en vrai c'est test_A qu'il faut plot, sinon on plot les images sur lesquelles on s'entraine, elles vont forcément être bien
if CONFIG['save_plots'] : save_plots (START_EPOCH-1, model, CONFIG['output_folder'], train_A, train_B, losses, CONFIG['dataset'])
if CONFIG['save_models'] : save_models (START_EPOCH-1, model, CONFIG['output_folder'], train_A, train_B, losses, CONFIG['dataset'])
    

# iterate through epochs
for i in range(START_EPOCH, CONFIG['end_epoch']):

    # initiate loss counter
    loss = []

    #get the number of batch
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
    
    # evaluate the model performance, and save
    #j'ai laissé le train_A dans le plot comme c'était, mais en vrai c'est test qu'il faut plot_A, sinon on plot les images sur lesquelles on s'entraine, elles vont forcément être bien
    if CONFIG['save_plots'] : save_plots (i, model, CONFIG['output_folder'], train_A, train_B, losses, CONFIG['dataset'])
    if CONFIG['save_models'] : save_models (i, model, CONFIG['output_folder'], train_A, train_B, losses, CONFIG['dataset'])