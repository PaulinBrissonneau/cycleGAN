#@title Data Visualization


from matplotlib import pyplot
from pylab import rcParams
from IPython.display import clear_output
import numpy as np


# probablement à réécrire pour pas avoir à réimporter tf ici, à discuter

def plot_sample(VIS_LINES, VIS_ROWS, PLOT_SIZE):

    #spec à écrire


    rcParams['figure.figsize'] = PLOT_SIZE, PLOT_SIZE

    # plot images from domain A dataset
    for i in range(VIS_LINES*VIS_ROWS):
        # define subplot
        pyplot.subplot(VIS_LINES, VIS_ROWS, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(tf.squeeze((train_A[i] + 1) / 2), cmap='gray_r')
    pyplot.show()

    # plot images from domain B dataset
    for i in range(VIS_LINES*VIS_ROWS):
        # define subplot
        pyplot.subplot(VIS_LINES, VIS_ROWS, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(tf.squeeze((train_B[i] + 1) / 2), cmap='gray_r')
    pyplot.show()


