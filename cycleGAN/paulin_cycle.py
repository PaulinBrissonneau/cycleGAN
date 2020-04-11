from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
from matplotlib import pyplot as plt
import numpy as np
import os
import datetime
from PIL import Image
from GAN import create_discriminator, create_generator

print("tf version : ", tf.__version__)

tf.config.experimental_run_functions_eagerly(True)

#passage en GPU
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

#PARAMS
EPOCHS = 1
BATCH = 1
LAMBD = 10
verbosity = 50
images_size =  (80, 64) #(80, 64)


###DATA

def get_data (batch_size, nb_train=None, nb_test=None) :

  direct_x = "/gpfs/workdir/brissonnp/celeb_smile_64_80/celebSmile/" #celeb_smile_168_216
  direct_y = "/gpfs/workdir/brissonnp/celeb_smile_64_80/celebNotSmile/"

  files_x = [direct_x + img for img in os.listdir(direct_x)[:30000]] #modifiable
  files_y = [direct_y + img for img in os.listdir(direct_y)[:30000]] #modifiable

  print("len(files_x) :",len(files_x))
  print("len(files_y) :",len(files_y))

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

G = create_generator() #X to Y
F = create_generator() #Y to X

print("models OK")

###TRAIN_STEP

@tf.function
def train_step(x_real_image, y_real_image, lambd):

    with tf.GradientTape() as Dx_tape, tf.GradientTape() as Dy_tape, tf.GradientTape() as G_tape, tf.GradientTape() as F_tape :

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        """couts :
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
        

        #verifications :
        #print(x_real_image.shape)
        #print(y_real_image.shape)
        #print(x_fake_image.shape)
        #print(y_fake_image.shape)
        #print(x_cycle_fake.shape)
        #print(y_cycle_fake.shape)

        def generator_loss(fake_output):
        
          return cross_entropy(tf.ones_like(fake_output), fake_output)

        def discriminator_loss(output_true, output_false):
        
          real_loss = cross_entropy(tf.ones_like(output_true)-0.01, output_true)

          fake_loss = cross_entropy(tf.zeros_like(output_false), output_false)

          total_loss = (real_loss+fake_loss)/2

          return total_loss

        G_loss_GAN = generator_loss(y_fake_output)
        Dy_loss_GAN = discriminator_loss(y_real_output, y_fake_output)

        F_loss_GAN = generator_loss(x_fake_output)
        Dx_loss_GAN = discriminator_loss(x_real_output, x_fake_output)

        loss_cycle = tf.reduce_mean(tf.abs(x_real_image-x_cycle_fake)) + tf.reduce_mean(tf.abs(y_real_image-y_cycle_fake))

        G_loss_total = G_loss_GAN+lambd*loss_cycle
        F_loss_total = F_loss_GAN+lambd*loss_cycle

        gradients_of_Dx = Dx_tape.gradient(Dx_loss_GAN, Dx['model'].trainable_variables)
        gradients_of_Dy = Dy_tape.gradient(Dy_loss_GAN, Dy['model'].trainable_variables)
        gradients_of_G = G_tape.gradient(G_loss_total, G['model'].trainable_variables)
        gradients_of_F = F_tape.gradient(F_loss_total, F['model'].trainable_variables)

        Dx['opti'].apply_gradients(zip(gradients_of_Dx, Dx['model'].trainable_variables))
        Dy['opti'].apply_gradients(zip(gradients_of_Dy, Dy['model'].trainable_variables))
        G['opti'].apply_gradients(zip(gradients_of_G, G['model'].trainable_variables))
        F['opti'].apply_gradients(zip(gradients_of_F, F['model'].trainable_variables))


    return Dx_loss_GAN, Dy_loss_GAN, G_loss_total, F_loss_total, x_real_image, x_fake_image, y_real_image, y_fake_image, x_cycle_fake, y_cycle_fake


###TRAIN

train_ds, test_ds = get_data(BATCH)

def train(dataset, epochs):

  L_Dx_loss = []
  L_Dy_loss = []
  L_G_loss = []
  L_F_loss = []
  X = []
  j = 0

  for epoch in range(epochs):
    
    now = datetime.datetime.now()
    dir_name = "cycleGAN-"+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"-"+str(now.hour)+"_"+str(now.minute)+"-ep:"+str(0).zfill(4)

    if not os.path.exists(dir_name):
      os.mkdir(dir_name)
      os.chdir(dir_name)
    else :
      raise Exception('Dossier existant')


    print("debut batch...")
    i = 0
    for image_1, image_2 in dataset:

        i += 1
        j += 1
        print(str(i)+"/"+str(30000/1)+" --- ") #modifier

        Dx_loss_GAN, Dy_loss_GAN, G_loss_total, F_loss_total, image_x, image_x_fake, image_y, image_y_fake, x_cycle_fake, y_cycle_fake = train_step(image_1, image_2, lambd=LAMBD)

        print("Dx_loss : "+str(round(Dx_loss_GAN.numpy(), 3))+" -- Dy_loss : "+str(round(Dy_loss_GAN.numpy(), 3))+" -- G_loss_total : "+str(round(G_loss_total.numpy(), 3))+" -- F_loss_total : "+str(round(F_loss_total.numpy(), 3)))

        if j % verbosity == 0:
          X.append(j)
          L_Dx_loss.append(Dx_loss_GAN)
          L_Dy_loss.append(Dy_loss_GAN)
          L_G_loss.append(G_loss_total)
          L_F_loss.append(F_loss_total)
          show_images (image_x, image_x_fake, image_y, image_y_fake, x_cycle_fake, y_cycle_fake, str(i))

          fig = plt.figure(figsize=(8,8))

          plt.subplot(2, 2, 1)
          plt.plot(X,L_Dx_loss, label = 'Dx_loss')
          plt.legend()
          plt.subplot(2, 2, 2)
          plt.plot(X,L_Dy_loss, label = 'Dy_loss')
          plt.legend()
          plt.subplot(2, 2, 3)
          plt.plot(X,L_G_loss, label = 'Gx_loss')
          plt.legend()
          plt.subplot(2, 2, 4)
          plt.plot(X,L_F_loss, label = 'Fy_loss')
          plt.legend()
          
          plt.savefig('cycleGAN_loss')

          plt.clf()
          plt.cla()
          plt.close()

          #saving
          call_G = G['model'].call.get_concrete_function(tf.TensorSpec((1, images_size[0], images_size[1], 3), tf.float32))
          model_path = "tf_saved_G_"+str(i)
          #tf.saved_model.save(G['model'], model_path, signatures=call_G) #juste pour tester
          call_F = F['model'].call.get_concrete_function(tf.TensorSpec((1, images_size[0], images_size[1], 3), tf.float32))
          model_path = "tf_saved_F_"+str(i)
          #tf.saved_model.save(F['model'], model_path, signatures=call_F) #juste pour tester

          os.chdir('..')
          new_dir_name = dir_name[:-4]+str(i).zfill(4)
          os.rename(dir_name, new_dir_name)
          os.chdir(new_dir_name)
          dir_name = new_dir_name

    print("fin batch...")


def show_images (image_x, image_x_fake, image_y, image_y_fake, x_cycle_fake, y_cycle_fake, epoch):

  L = [image_x, image_x_fake, image_y, image_y_fake, x_cycle_fake, y_cycle_fake]
  Llab = ["image_x", "image_y_fake_x", "image_y", "image_x_fake_y", "image_x_cycle", "image_y_cycle"]

  ###PLT

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
  
  plt.savefig('cycleGAN_at_epoch_'+epoch+'.png')
  plt.clf()
  plt.cla()
  plt.close()

  #IMAGE (+qualité)

  for k in range(len(L)):

    file_name = 'cycleGAN_at_epoch_'+epoch+'_'+Llab[k]+'.png'
    im = np.clip((L[k].numpy()[0]+(1/2))*255, 0, 255)
    im = Image.fromarray(im.astype(np.uint8))
    im.save(file_name) 


train(train_ds, EPOCHS)