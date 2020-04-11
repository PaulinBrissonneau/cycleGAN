from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
from matplotlib import pyplot as plt
import numpy as np
import os
import datetime
from PIL import Image
import random as rd
from GANres import create_discriminator, create_generator

print("tf version : ", tf.__version__)

tf.config.experimental_run_functions_eagerly(True)

#passage en GPU
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

#PARAMS
EPOCHS = 200
BATCH = 1
LAMBD_CYCLE = 10
LAMBD_ID = 5
verbosity = 50
MAX_BUFFER_SIZE = 50
images_size =  (128, 128) #(80, 64)

###DATA

def get_data (batch_size, nb_train=None, nb_test=None) :

  direct_x = "/gpfs/workdir/brissonnp/orangeApple/orange/" #celeb_smile_168_216
  direct_y = "/gpfs/workdir/brissonnp/orangeApple/apple/"

  files_x = [direct_x + img for img in os.listdir(direct_x)]
  files_y = [direct_y + img for img in os.listdir(direct_y)]

  nb_x = len(files_x)
  nb_y = len(files_y)

  print("len(files_x) :",nb_x)
  print("len(files_y) :",nb_y)

  filenames_x = tf.constant(files_x)
  filenames_y = tf.constant(files_y)

  dataset = tf.data.Dataset.from_tensor_slices((filenames_x, filenames_y))

  def _parse_function(filenames_x, filenames_y):
    
    image_string_x = tf.io.read_file(filenames_x)
    image_string_y = tf.io.read_file(filenames_y)

    image_decoded_x = tf.image.decode_jpeg(image_string_x, channels=3)
    image_decoded_y = tf.image.decode_jpeg(image_string_y, channels=3)

    image_x = tf.cast(image_decoded_x, tf.float32)
    image_y = tf.cast(image_decoded_y, tf.float32)

    image_x = (image_x-(255/2))/255
    image_y = (image_y-(255/2))/255

    return image_x, image_y

  dataset = dataset.map(_parse_function)
  train_ds = dataset.batch(BATCH)

  test_ds = None

  return train_ds, test_ds


###MODELS

Dx = create_discriminator(_input_shape = (None, images_size[0], images_size[1], 3))
Dy = create_discriminator(_input_shape = (None, images_size[0], images_size[1], 3))

G = create_generator(_n_resnet = 6) #X to Y
F = create_generator(_n_resnet = 6) #Y to X

print("models OK")

###TRAIN_STEP

@tf.function
def train_step(x_real_image, y_real_image, buffer_x, buffer_y, lambd_id, lambd_cycle):

    with tf.GradientTape() as Dx_tape, tf.GradientTape() as Dy_tape, tf.GradientTape() as G_tape, tf.GradientTape() as F_tape :

        #cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        """couts : NOPE !!
          mean(log(Dy(y)))+mean(log(1-Dy(G(x)))) sur Dy et G
          mean(log(Dx(x)))+mean(log(1-Dx(F(y)))) sur Dx et F
          mean(L1(F(G(x))-x))+mean(L1(G(F(y)))) sur G et F"""

        
        x_fake_image = F["model"](y_real_image)
        y_fake_image = G["model"](x_real_image)
        
        x_fake_output = Dx['model'](x_fake_image)
        x_real_output = Dx['model'](x_real_image)

        y_fake_output = Dy['model'](y_fake_image)
        y_real_output = Dy['model'](y_real_image)

        x_cycle_fake = F['model'](y_fake_image)
        y_cycle_fake = G['model'](x_fake_image)
        
        """
        #verifications :
        print(x_real_image.shape)
        print(y_real_image.shape)
        print(x_fake_image.shape)
        print(y_fake_image.shape)
        print(x_cycle_fake.shape)
        print(y_cycle_fake.shape)

        print(x_fake_output.shape)
        print(x_real_output.shape)

        print(y_fake_output.shape)
        print(y_real_output.shape)"""

        def generator_loss(fake_output):
        
          return  tf.reduce_mean(tf.keras.losses.MSE(tf.ones_like(fake_output), fake_output))

        def discriminator_loss(output_true, output_false):
        
          real_loss = tf.reduce_mean(tf.keras.losses.MSE(tf.ones_like(output_true)-0.01, output_true))

          fake_loss = tf.reduce_mean(tf.keras.losses.MSE(tf.zeros_like(output_false), output_false))

          total_loss = (real_loss+fake_loss)/2

          return total_loss

        def cycle_loss (real, cycle_fake):

          return tf.reduce_mean(tf.keras.losses.MAE(real, cycle_fake))

        def cycle_id (real, fake_next):

          return tf.reduce_mean(tf.keras.losses.MAE(real, fake_next))

        G_loss_GAN = generator_loss(y_fake_output)
        Dy_loss_GAN = discriminator_loss(y_real_output, buffer_y.get_image(y_fake_output))

        F_loss_GAN = generator_loss(x_fake_output)
        Dx_loss_GAN = discriminator_loss(x_real_output, buffer_x.get_image(x_fake_output))


        loss_cycle = cycle_loss(x_real_image, x_cycle_fake) + cycle_loss(y_real_image, y_cycle_fake)

        loss_identity = cycle_id (x_real_image, y_fake_image) + cycle_id (y_real_image, x_fake_image)

        GF_loss_total = lambd_id*loss_identity+lambd_cycle*loss_cycle+G_loss_GAN+F_loss_GAN

        gradients_of_Dx = Dx_tape.gradient(Dx_loss_GAN, Dx['model'].trainable_variables)
        gradients_of_Dy = Dy_tape.gradient(Dy_loss_GAN, Dy['model'].trainable_variables)
        gradients_of_G = G_tape.gradient(GF_loss_total, G['model'].trainable_variables)
        gradients_of_F = F_tape.gradient(GF_loss_total, F['model'].trainable_variables)

        Dx['opti'].apply_gradients(zip(gradients_of_Dx, Dx['model'].trainable_variables))
        Dy['opti'].apply_gradients(zip(gradients_of_Dy, Dy['model'].trainable_variables))
        G['opti'].apply_gradients(zip(gradients_of_G, G['model'].trainable_variables))
        F['opti'].apply_gradients(zip(gradients_of_F, F['model'].trainable_variables))


    return Dx_loss_GAN, Dy_loss_GAN, loss_identity, loss_cycle, G_loss_GAN, F_loss_GAN, GF_loss_total, x_real_image, x_fake_image, y_real_image, y_fake_image, x_cycle_fake, y_cycle_fake


###TRAIN

train_ds, test_ds = get_data(BATCH)

def train(dataset, epochs):

  nb= len(list(dataset))

  L_Dx_loss = []
  L_Dy_loss = []
  L_loss_identity = []
  L_loss_cycle = []
  L_G_loss_GAN = []
  L_F_loss_GAN = []
  L_GF_loss_total = []
  X = []
  j = 0

  now = datetime.datetime.now()
  workdir_name = "/gpfs/workdir/brissonnp/output/"
  dir_name = "cycleOrange-"+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"-"+str(now.hour)+"_"+str(now.minute)

  if not os.path.exists(workdir_name+dir_name):
    os.mkdir(workdir_name+dir_name)
    os.chdir(workdir_name+dir_name)
  else :
    raise Exception('Dossier existant')

  for epoch in range(epochs):

    buffer_x = Buffer(MAX_BUFFER_SIZE)
    buffer_y = Buffer(MAX_BUFFER_SIZE)

    print("debut batch...")
    i = 0
    for image_1, image_2 in dataset:

        i += 1
        j += 1
        
        print(str(i)+"/"+str(nb/BATCH)+" --- ")

        Dx_loss_GAN, Dy_loss_GAN, loss_identity, loss_cycle, G_loss_GAN, F_loss_GAN, GF_loss_total, image_x, image_x_fake, image_y, image_y_fake, x_cycle_fake, y_cycle_fake = train_step(image_1, image_2, buffer_x, buffer_y, lambd_id=LAMBD_ID, lambd_cycle=LAMBD_CYCLE)

        buffer_x.update (image_x_fake)
        buffer_y.update (image_y_fake)

        print("Dx_loss : "+str(round(Dx_loss_GAN.numpy(), 3))+" -- Dy_loss : "+str(round(Dy_loss_GAN.numpy(), 3))+" -- GF_loss_total : "+str(round(GF_loss_total.numpy(), 3)))

        if j % verbosity == 0:
          X.append(j)
          L_Dx_loss.append(Dx_loss_GAN)
          L_Dy_loss.append(Dy_loss_GAN)
          L_loss_identity.append(loss_identity)
          L_loss_cycle.append(loss_cycle)
          L_G_loss_GAN.append(G_loss_GAN)
          L_F_loss_GAN.append(F_loss_GAN)
          L_GF_loss_total.append(GF_loss_total)
          show_images (image_x, image_x_fake, image_y, image_y_fake, x_cycle_fake, y_cycle_fake, str(epoch).zfill(6), str(i).zfill(6))

          fig = plt.figure(figsize=(16,8))

          plt.subplot(2, 4, 1)
          plt.plot(X,L_Dx_loss, label = 'Dx_loss')
          plt.legend()
          plt.subplot(2, 4, 2)
          plt.plot(X,L_Dy_loss, label = 'Dy_loss')
          plt.legend()
          plt.subplot(2, 4, 3)
          plt.plot(X,L_loss_identity, label = 'loss_identity')
          plt.legend()
          plt.subplot(2, 4, 4)
          plt.plot(X,L_loss_cycle, label = 'loss_cycle')
          plt.legend()
          plt.subplot(2, 4, 5)
          plt.plot(X,L_G_loss_GAN, label = 'G_loss_GAN')
          plt.legend()
          plt.subplot(2, 4, 6)
          plt.plot(X,L_F_loss_GAN, label = 'F_loss_GAN')
          plt.legend()
          plt.subplot(2, 4, 7)
          plt.plot(X,L_GF_loss_total, label = 'GF_loss_total')
          plt.legend()
          
          plt.savefig('cycleGAN_loss')
          print("saved")

          plt.clf()
          plt.cla()
          plt.close()

          #saving
          call_G = G['model'].call.get_concrete_function(tf.TensorSpec((1, images_size[0], images_size[1], 3), tf.float32))
          model_path = "tf_saved_G_"+str(j)
          #tf.saved_model.save(G['model'], model_path, signatures=call_G) #juste pour tester
          call_F = F['model'].call.get_concrete_function(tf.TensorSpec((1, images_size[0], images_size[1], 3), tf.float32))
          model_path = "tf_saved_F_"+str(j)
          #tf.saved_model.save(F['model'], model_path, signatures=call_F) #juste pour tester

          """
          os.chdir('..')
          new_dir_name = dir_name
          os.rename(dir_name, new_dir_name)
          os.chdir(new_dir_name)
          dir_name = new_dir_name"""

    print("fin batch...")


class Buffer :

  def __init__(self, max_size):
    self.max_size = max_size
    self.pool = []

  def get_image (self, new_image):
    if len(self.pool)== 0:
      return new_image
    if rd.random() < 0.5 :
      return new_image
    else :
      return rd.choice(self.pool)

  def update (self, image):
    if len(self.pool) < self.max_size :
      self.pool.append(image)
    else:
      pop = rd.randint(0, len(self.pool)-1)
      self.pool.pop(pop)
      self.pool.append(image)


def show_images (image_x, image_x_fake, image_y, image_y_fake, x_cycle_fake, y_cycle_fake, epoch, batch):

  L = [image_x, image_x_fake, image_y, image_y_fake, x_cycle_fake, y_cycle_fake]
  Llab = ["image_x", "image_y_fake_x", "image_y", "image_x_fake_y", "image_x_cycle", "image_y_cycle"]

  ###PLT

  name_save = 'cycleGAN_epoch_'+epoch+'_batch_'+batch

  fig = plt.figure(figsize=(3,3))

  plt.subplot(3, 3, 1)
  plt.imshow((image_x[0]+1)/2)
  plt.subplot(3, 3, 2)
  plt.imshow((image_y_fake[0]+1)/2)
  plt.subplot(3, 3, 3)
  plt.imshow((x_cycle_fake[0]+1)/2)
  plt.subplot(3, 3, 7)
  plt.imshow((image_y[0]+1)/2)
  plt.subplot(3, 3, 8)
  plt.imshow((image_x_fake[0]+1)/2)
  plt.subplot(3, 3, 9)
  plt.imshow((y_cycle_fake[0]+1)/2)
  plt.axis('off')
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
  
  plt.savefig(name_save+'.png')
  plt.clf()
  plt.cla()
  plt.close()

  #IMAGE (+qualité)

  for k in range(len(L)):

    file_name = name_save+'_'+Llab[k]+'.png'
    im = np.clip((L[k].numpy()[0]+(1/2))*255, 0, 255)
    im = Image.fromarray(im.astype(np.uint8))
    im.save(file_name) 


train(train_ds, EPOCHS)