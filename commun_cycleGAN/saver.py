from IPython.display import clear_output
import numpy as np

N_SAMPLES = 3 #@param {type:"integer"}

# create and save a plot of generated images
def save_plot(epoch, model, test_X, domains, n_samples=N_SAMPLES):
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
def save_model(epoch, model, component):
    file_name = f'models/{DATASET}/cycleGAN_e{epoch+1:03}_{component}'
    model.save(file_name)

def save_history(losses):
    # plot history
    pyplot.plot(losses.A_to_B_loss.values, label='A_to_B_loss')
    pyplot.plot(losses.B_to_A_loss.values, label='B_to_A_loss')
    pyplot.legend()
    file_name = f'plots/{DATASET}/cycleGAN_loss_history.png'
    pyplot.savefig(file_name)
    pyplot.close()