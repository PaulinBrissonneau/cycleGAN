# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
from config_reader import *
from data_process import *
from display import *
from saver import *
from builder import *

LOAD_MODEL = read_config("LOAD_MODEL")

#à ajouter dans le config reader :
END_EPOCH = 100  # @param {type:"integer"}
BATCH_SIZE = 1  # @param {type:"integer"}
TEST_RATIO = 0.2

if LOAD_MODEL:
    START_EPOCH = LOAD_EPOCH
else:
    START_EPOCH = 0

# pour moi  TRAIN_A_BUF  TRAIN_B_BUF c'est à enlever, c'est pas nécessaire, comme vous voulez
#train_A, train_B, test_A, test_B, DIMS, DATASET = get_datas ("vangogh2photo")
train_A, train_B, test_A, test_B, DIMS, DATASET, TRAIN_A_BUF, TRAIN_B_BUF = get_datas_paulin("vangogh2photo", test_ratio = TEST_RATIO)

#pas de TEST_A_BUF car pas de test dans le code de base, il faudra rajouter

TRAIN_BUF = min([TRAIN_A_BUF, TRAIN_B_BUF])
TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)

# a pandas dataframe to save the loss information to
losses = DataFrame(columns = ['A_to_B_loss', 'B_to_A_loss'])
losses.loc[len(losses)] = (0, 0)

# ajouter ca dans config_params
VIS_LINES = 1 #@param {type:"integer"}
VIS_ROWS = 3 #@param {type:"integer"}

PLOT_SIZE = 20 #@param {type:"integer"}

#pour l'instant incompatible
#plot_sample(VIS_LINES, VIS_ROWS, PLOT_SIZE)

# ajouter ca dans config_params
ALPHA = 0.0002 #@param {type:"number"}
BETA_1 = 0.5 #@param {type:"number"}

model = build_cycleGAN(ALPHA, BETA_1, DIMS, DATASET)

#pour l'instant incompatible
#save_performance(START_EPOCH-1, model, train_A, train_B, losses, DATASET)

pool_A, pool_B = [], []

train_A, train_B, test_A, test_B = train_A.batch(BATCH_SIZE), train_B.batch(BATCH_SIZE), test_A.batch(BATCH_SIZE), test_B.batch(BATCH_SIZE)

# iterate through epochs
for i in range(START_EPOCH, END_EPOCH):
    # initiate loss counter
    loss = []

    #enumerate batches over the training set
    j = 0
    for image_1, image_2 in zip(train_A, train_B) :
        j+= 1
        
        real_a = image_1
        real_b = image_2

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
    