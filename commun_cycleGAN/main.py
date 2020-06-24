# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import tensorflow as tf
from pathlib import Path
import datetime

from config_reader import *
from data_process import *
from display import *
from saver import *
from cycleGAN_builder import *
import sys


#passage en eager execution
tf.config.experimental_run_functions_eagerly(True)

#read configuration file
config_file = sys.argv[1]
print("config_file : ", config_file)
CONFIG = read_config(config_file)

#passage en GPU
if CONFIG['on_gpu'] :
    print("nombre de GPU : ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)

train_A, train_B, test_A, test_B, DIMS = get_datas_mapping(test_ratio = CONFIG['test_ratio'], data_x_folder = CONFIG['data_x_folder'], data_y_folder = CONFIG['data_y_folder'], debug_sample=CONFIG['debug_sample'])

#prevent collisions in directory names
now = datetime.datetime.now()
now = "_"+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"-"+str(now.hour)+"_"+str(now.minute)
output_folder_date = 'output'+now
PATHS = [
            f"{CONFIG['name']}/{output_folder_date}/plots/",
            f"{CONFIG['name']}/{output_folder_date}/models/",
            f"{CONFIG['name']}/{output_folder_date}/logs/" 
        ]

for path in PATHS:
    Path(path).mkdir(parents=True, exist_ok=True)


# a pandas dataframe to save the loss information to
losses = DataFrame(columns = ['A_to_B_loss', 'B_to_A_loss'])
losses.loc[len(losses)] = (0, 0)

# display sample / visualisation
if CONFIG['plot_sample']: plot_sample(train_A, train_B, CONFIG['vis_lines'], CONFIG['vis_rows'], CONFIG['plot_size'])

model = build_cycleGAN(CONFIG['alpha'], CONFIG['beta_1'], DIMS, CONFIG['dataset'], CONFIG['max_buffer_size'], CONFIG['n_resnet'])



#make batches from the tf.Dataset
BATCH_SIZE = CONFIG['batch_size']
train_A, train_B, test_A, test_B = train_A.batch(BATCH_SIZE), train_B.batch(BATCH_SIZE), test_A.batch(BATCH_SIZE), test_B.batch(BATCH_SIZE)

#check if existing model
last_epoch, models_folder, last_qsub = restore_epoch (CONFIG['name'])


#continue training if existing model
if last_epoch > 0 :
    qsub = last_qsub+1
    START_EPOCH = last_epoch+1
    load_weights(CONFIG['name'], model, models_folder, last_epoch)
    print("Model loaded - starting fit...")
else :
    START_EPOCH = 1
    qsub = 1

#create text file for Fusion checkpoints
create_checkpoint(qsub, CONFIG['name'], START_EPOCH, output_folder_date)

#plots
if CONFIG['save_plots'] : save_plots (CONFIG['name'], START_EPOCH, model, output_folder_date, train_A, train_B, losses, CONFIG['n_sample'], comment = '_STARTING')
if CONFIG['save_models'] : save_models (CONFIG['name'], START_EPOCH, model, output_folder_date, train_A, train_B, losses)

# iterate through epochs
for i in range(START_EPOCH, START_EPOCH+CONFIG['number_of_epochs']):

    # initiate loss counter
    loss = []

    #get the number of batch
    number_of_batch = len(list(zip(train_A, train_B)))

    #update learning rate
    current_learning_rate = tf.compat.v1.train.polynomial_decay(learning_rate=CONFIG['alpha'], global_step=i, decay_steps=CONFIG['decay_steps'], end_learning_rate=CONFIG['end_learning_rate'], power=1.0)
    model.disc_A_optimizer.learning_rate = current_learning_rate
    model.disc_B_optimizer.learning_rate = current_learning_rate
    model.gen_A_to_B_optimizer.learning_rate = current_learning_rate
    model.gen_B_to_A_optimizer.learning_rate = current_learning_rate

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

        # diplay during batch
        if CONFIG['save_plots_during_batch'] and int(tqdm_bar.n)%int(CONFIG['freq_plots_during_batch']) == 0 : save_plots (CONFIG['name'], i, model, output_folder_date, train_A, train_B, losses, CONFIG['n_sample'], during_batch = True, batch = int(tqdm_bar.n))

    # average loss over epoch
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    
    # evaluate the model performance, and save
    #j'ai laissé le train_A dans le plot comme c'était, mais en vrai c'est test qu'il faut plot_A, sinon on plot les images sur lesquelles on s'entraine, elles vont forcément être bien
    if CONFIG['save_plots'] : save_plots (CONFIG['name'], i, model, output_folder_date, train_A, train_B, losses, CONFIG['n_sample'])
    if CONFIG['save_models'] : save_models (CONFIG['name'], i, model, output_folder_date, train_A, train_B, losses)

    update_checkpoint(CONFIG['name'], i)