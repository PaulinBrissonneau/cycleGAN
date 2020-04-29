from IPython.display import clear_output
import numpy as np
from matplotlib import pyplot
import tensorflow as tf

# evaluate the discrimenator, plot generated images
def save_plots (epoch, model, output_folder, test_A, test_B, losses, DATASET, n_samples,  during_batch = False, batch = -1):
    # a simple summerizing print
    print(f"Epoch: {epoch+1} | A_to_B_loss: {losses.A_to_B_loss.values[-1]} | B_to_A_loss: {losses.B_to_A_loss.values[-1]}")
    # save plot
    save_plot(epoch, model.A_to_B, output_folder, test_A, 'A_to_B', DATASET, n_samples, during_batch, batch)
    save_plot(epoch, model.B_to_A, output_folder, test_B, 'B_to_A', DATASET, n_samples, during_batch, batch)
    # save history
    save_history(losses, output_folder, DATASET)


#save generator model and discriminators
def save_models(epoch, model, output_folder, test_A, test_B, losses, DATASET):
    # save the model
    save_model(epoch, model.disc_A, output_folder, 'disc_A', DATASET)
    save_model(epoch, model.disc_B, output_folder, 'disc_B', DATASET)
    save_model(epoch, model.gen_A_to_B, output_folder, 'gen_A_to_B', DATASET)
    save_model(epoch, model.gen_B_to_A, output_folder,'gen_B_to_A', DATASET)



# create and save a plot of generated images
def save_plot(epoch, model, output_folder, test_X, domains, DATASET, n_samples, during_batch, batch):
    # choose random images from dataset
    ix = list(np.random.randint(0, len(list(test_X)), n_samples))
    #transformation from tf.data to numpy array
    test_x = np.array([np.array(list(test_X)[i][0]) for i in ix])
    # generate transfered images
    gen_x = model(test_x)
    # scale from [-1,1] to [0,1] and clip()
    test_x = np.clip(((test_x + 1.) / 2.), 0, 1)
    gen_x = np.clip((gen_x + 1.) / 2., 0, 1)

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
    if during_batch : file_name = f"{output_folder}/plots/{DATASET}/cycleGAN_epoch{epoch+1:03}_batch{batch}_{domains}.png"
    else : file_name = f"{output_folder}/plots/{DATASET}/cycleGAN_epoch{epoch+1:03}_{domains}.png"
    pyplot.savefig(file_name)
    # pyplot.show()
    pyplot.close()

# save the whole model if possible xD
def save_model(epoch, model, output_folder, component, DATASET):
    file_name = f"{output_folder}/models/{DATASET}/cycleGAN_e{epoch+1:03}_{component}"
    model.save(file_name)

def save_history(losses,output_folder, DATASET):
    # plot history
    pyplot.plot(losses.A_to_B_loss.values, label='A_to_B_loss')
    pyplot.plot(losses.B_to_A_loss.values, label='B_to_A_loss')
    pyplot.legend()
    file_name = f"{output_folder}/plots/{DATASET}/cycleGAN_loss_history.png"
    pyplot.savefig(file_name)
    pyplot.close()