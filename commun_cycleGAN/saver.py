from IPython.display import clear_output
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
import os.path

# evaluate the discrimenator, plot generated images
def save_plots (name, epoch, model, output_folder, test_A, test_B, losses, n_samples,  during_batch = False, batch = -1, comment = ''):
    # a simple summerizing print
    print(f"Epoch: {epoch} | A_to_B_loss: {losses.A_to_B_loss.values[-1]} | B_to_A_loss: {losses.B_to_A_loss.values[-1]}")
    # save plot
    save_plot(name, epoch, model.A_to_B, output_folder, test_A, 'A_to_B'+comment, n_samples, during_batch, batch)
    save_plot(name, epoch, model.B_to_A, output_folder, test_B, 'B_to_A'+comment, n_samples, during_batch, batch)
    # save history
    save_history(name, losses, output_folder)


#save generator model and discriminators
def save_models(name, epoch, model, output_folder, test_A, test_B, losses):
    # save the model
    save_model(name, epoch, model.disc_A, output_folder, 'disc_A')
    save_model(name, epoch, model.disc_B, output_folder, 'disc_B')
    save_model(name, epoch, model.gen_A_to_B, output_folder, 'gen_A_to_B')
    save_model(name, epoch, model.gen_B_to_A, output_folder,'gen_B_to_A')

def load_weights(name, model, output_folder, START_EPOCH):

    def load_one (name, epoch, submodel, output_folder, component):
        file_name = f"{name}/{output_folder}/models/cycleGAN_e{epoch:03}_{component}"
        submodel.load_weights(file_name)
        #submodel = tf.keras.models.load_model(file_name)
        
    load_one (name, START_EPOCH, model.disc_A, output_folder, 'disc_A')
    load_one (name, START_EPOCH, model.disc_B, output_folder, 'disc_B')
    load_one (name, START_EPOCH, model.gen_A_to_B, output_folder, 'gen_A_to_B')
    load_one (name, START_EPOCH, model.gen_B_to_A, output_folder, 'gen_B_to_A')


# create and save a plot of generated images
def save_plot(name, epoch, model, output_folder, test_X, domains, n_samples, during_batch, batch):
    # choose random images from dataset
    ix = list(np.random.randint(0, len(list(test_X)), n_samples))
    #transformation from tf.data to numpy array
    test_x = np.array([np.array(list(test_X)[i][0]) for i in ix])
    # generate transfered images
    gen_x = model(test_x) #checked : in ]0, 1[

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
    if during_batch : file_name = f"{name}/{output_folder}/plots/cycleGAN_epoch{epoch:03}_batch{batch}_{domains}.png"
    else : file_name = f"{name}/{output_folder}/plots/cycleGAN_epoch{epoch:03}_{domains}.png"
    pyplot.savefig(file_name)
    # pyplot.show()
    pyplot.close()

# save the whole model if possible xD
def save_model(name, epoch, model, output_folder, component):
    file_name = f"{name}/{output_folder}/models/cycleGAN_e{epoch:03}_{component}"
    #model.save(file_name)
    model.save_weights(file_name)

def save_history(name, losses,output_folder):
    # plot history
    pyplot.plot(losses.A_to_B_loss.values, label='A_to_B_loss')
    pyplot.plot(losses.B_to_A_loss.values, label='B_to_A_loss')
    pyplot.legend()
    file_name = f"{name}/{output_folder}/plots/cycleGAN_loss_history.png"
    pyplot.savefig(file_name)
    pyplot.close()


#create the txt checkpoint file
def create_checkpoint (i, name, epoch, folder) :
    f= open(f"{name}/checkpoints.txt","a")
    f.write(f"QSUB {i} :\n")
    f.write(f"  Start epoch : {epoch:03}\n")
    f.write(f"  Output folder : {folder}\n")
    f.write(f"  Last epoch : {epoch:03}\n")
    f.write(f"\n")
    f.close()
    return None

#change last lines of the checkpoints
def update_checkpoint (name, epoch) :
    f= open(f"{name}/checkpoints.txt","r+")
    lines = f.readlines()
    f.close()
    previous = lines[:-2]
    f= open(f"{name}/checkpoints.txt","w+")
    f.write(''.join(previous))
    f.write(f"  Last epoch : {epoch:03}\n")
    f.write(f"\n")
    f.close()
    return None


#check if the file is empty, if not, find last epoch and model folders
def restore_epoch (name) :

    #if no model yet
    if not os.path.isfile(f"{name}/checkpoints.txt") :
        last_epoch = 0
        models_folder = "not defined"
        last_qsub = 0

        print("\n")
        print("--- NO MODEL TO LOAD - starting from scratch ---")
        print("\n")

    #load model if exists 
    else :
        f= open(f"{name}/checkpoints.txt","r+")
        lines = f.readlines()
        f.close()
        last_epoch = int(lines[-2].split(" : ")[-1])
        models_folder = str(lines[-3].split(" : ")[-1]).replace("\n", "").replace(" ", "")
        last_qsub = int(lines[-5].split(" ")[1])
        
        #recap
        print("\n")
        print("--- MODEL TO LOAD ---")
        print("last_qsub : ", last_qsub)
        print("last_epoch : ", last_epoch)
        print("models_folder : ", models_folder)
        print("\n")

    return last_epoch, models_folder, last_qsub