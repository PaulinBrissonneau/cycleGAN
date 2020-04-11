from IPython.display import clear_output
import numpy as np
from matplotlib import pyplot

#à passer dans le main, dans config, etc
N_SAMPLES = 3 #@param {type:"integer"}


# evaluate the discrimenator, plot generated images, save generator model
def save_performance(epoch, model, test_A, test_B, losses, DATASET):
    # a simple summerizing print
    print(f"Epoch: {epoch+1} | A_to_B_loss: {losses.A_to_B_loss.values[-1]} | B_to_A_loss: {losses.B_to_A_loss.values[-1]}")
    # save plot
    save_plot(epoch, model.A_to_B, test_A, 'A_to_B', DATASET)
    save_plot(epoch, model.B_to_A, test_B, 'B_to_A', DATASET)
    # save history
    save_history(losses, DATASET)
    # save the model
    save_model(epoch, model.disc_A, 'disc_A', DATASET)
    save_model(epoch, model.disc_B, 'disc_B', DATASET)
    save_model(epoch, model.gen_A_to_B, 'gen_A_to_B', DATASET)
    save_model(epoch, model.gen_B_to_A, 'gen_B_to_A', DATASET)



# create and save a plot of generated images
def save_plot(epoch, model, test_X, domains, DATASET,  n_samples=N_SAMPLES):
    # choose random images from dataset
    ix = np.random.randint(0, len(test_X), n_samples)
    test_x = test_X[ix]
    # generte trasfered images
    gen_x = model(test_x)
    # scale from [-1,1] to [0,1]
    test_x = (test_x + 1.) / 2.
    gen_x = (gen_x + 1.) / 2.
    # plot real images
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(test_x[i])
    # plot translated image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(gen_x[i])
    # save plot to file
    file_name = f'plots/{DATASET}/cycleGAN_e{epoch+1:03}_{domains}.png'
    pyplot.savefig(file_name)
    # pyplot.show()
    pyplot.close()

# save the whole model if possible xD
def save_model(epoch, model, component, DATASET):
    file_name = f'models/{DATASET}/cycleGAN_e{epoch+1:03}_{component}'
    model.save(file_name)

def save_history(losses, DATASET):
    # plot history
    pyplot.plot(losses.A_to_B_loss.values, label='A_to_B_loss')
    pyplot.plot(losses.B_to_A_loss.values, label='B_to_A_loss')
    pyplot.legend()
    file_name = f'plots/{DATASET}/cycleGAN_loss_history.png'
    pyplot.savefig(file_name)
    pyplot.close()