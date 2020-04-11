#@title Data Visualization


from matplotlib import pyplot
from pylab import rcParams

VIS_LINES = 1 #@param {type:"integer"}
VIS_ROWS = 3 #@param {type:"integer"}

PLOT_SIZE = 20 #@param {type:"integer"}

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



from IPython.display import clear_output
import numpy as np

N_SAMPLES = 3 #@param {type:"integer"}

# evaluate the discrimenator, plot generated images, save generator model
def summarize_performance(epoch, model, test_A, test_B, losses):
    # a simple summerizing print
    print(f"Epoch: {epoch+1} | A_to_B_loss: {losses.A_to_B_loss.values[-1]} | B_to_A_loss: {losses.B_to_A_loss.values[-1]}")
    # save plot
    save_plot(epoch, model.A_to_B, test_A, 'A_to_B')
    save_plot(epoch, model.B_to_A, test_B, 'B_to_A')
    # save history
    save_history(losses)
    # save the model
    save_model(epoch, model.disc_A, 'disc_A')
    save_model(epoch, model.disc_B, 'disc_B')
    save_model(epoch, model.gen_A_to_B, 'gen_A_to_B')
    save_model(epoch, model.gen_B_to_A, 'gen_B_to_A')