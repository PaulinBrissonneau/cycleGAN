# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from pandas import DataFrame

if LOAD_MODEL:
    START_EPOCH = LOAD_EPOCH
else:
    START_EPOCH = 0

END_EPOCH = 100  # @param {type:"integer"}
BATCH_SIZE = 1  # @param {type:"integer"}


TRAIN_A_BUF = len(train_A)
TRAIN_B_BUF = len(train_B)
TRAIN_BUF = min([TRAIN_A_BUF, TRAIN_B_BUF])
TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)

# a pandas dataframe to save the loss information to
losses = DataFrame(columns = ['A_to_B_loss', 'B_to_A_loss'])
losses.loc[len(losses)] = (0, 0)
summarize_performance(START_EPOCH-1, model, train_A, train_B, losses)

pool_A, pool_B = [], []

# iterate through epochs
for i in range(START_EPOCH, END_EPOCH):
    # initiate loss counter
    loss = []
    # enumerate batches over the training set
    for j in tqdm(range(TRAIN_BATCHES)):
        # take random indexes
        ia = np.random.randint(0, TRAIN_A_BUF, BATCH_SIZE)
        ib = np.random.randint(0, TRAIN_B_BUF, BATCH_SIZE)
        # select images
        real_a = train_A[ia]
        real_b = train_B[ib]
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
    # clear previous results
    clear_output()
    # evaluate the model performance, sometimes
    summarize_performance(i, model, train_A, train_B, losses)