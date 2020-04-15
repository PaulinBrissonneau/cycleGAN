#@title Data Visualization


from matplotlib import pyplot
from pylab import rcParams
from IPython.display import clear_output
import numpy as np


# probablement à réécrire pour pas avoir à réimporter tf ici, pas opti du tout, à discuter
#import tensorflow as tf

def plot_sample(train_A, train_B, VIS_LINES, VIS_ROWS, PLOT_SIZE, ):

    #in : plot params
    #out : none

    rcParams['figure.figsize'] = PLOT_SIZE, PLOT_SIZE

    #get a sample from our databases
    train_A = list(train_A.take(VIS_LINES*VIS_ROWS).as_numpy_iterator())
    train_B = list(train_B.take(VIS_LINES*VIS_ROWS).as_numpy_iterator())
   

    # plot images from domain A dataset
    for i in range(VIS_LINES*VIS_ROWS):
        # define subplot
        pyplot.subplot(VIS_LINES, VIS_ROWS, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow((train_A[i] + 1) / 2, cmap='gray_r')
    pyplot.show()

    # plot images from domain B dataset
    for i in range(VIS_LINES*VIS_ROWS):
        # define subplot
        pyplot.subplot(VIS_LINES, VIS_ROWS, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow((train_B[i] + 1) / 2, cmap='gray_r')
    pyplot.show()