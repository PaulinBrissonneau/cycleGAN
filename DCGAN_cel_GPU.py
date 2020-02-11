from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import time
from matplotlib import pyplot as plt
import numpy as np
import os
import datetime

print("tf version : ", tf.__version__)

#gpu = tf.config.experimental.list_physical_devices('GPU')[0]
#tf.config.experimental.set_memory_growth(gpu, True)

def get_data (batch_size, nb_train, nb_test) :

  direct = "/home/paulin/Documents/datas/celeb/celebresize/"
files = [direct+str(i).zfill(6)+".jpg" for i in range (1, nb_train)]

print(len(files))

filenames = tf.constant(files)

dataset = tf.data.Dataset.from_tensor_slices((filenames))

  def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image = (image-(255/2))/255
    return image

  dataset = dataset.map(_parse_function)
  train_ds = dataset.shuffle(10000).batch(batch_size)

  return train_ds, test_ds=None


class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    self.dense0 = tf.keras.layers.Dense(3*10*8, use_bias=False, input_shape=(100,))
    self.acti0 = tf.keras.layers.ReLU()

    self.resh01 = tf.keras.layers.Reshape((10, 8, 3))

    self.conv1 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.acti1 = tf.keras.layers.ReLU()
    #shape 10x8

    self.conv2 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.acti2 = tf.keras.layers.ReLU()
    #shape 20x16

    self.conv3 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.acti3 = tf.keras.layers.ReLU()
    #shape 40x32

    self.conv4 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    #shape 80*64

  def call(self, x):
    x = self.dense0(x)
    x = self.acti0(x)
    x = self.resh01(x)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.acti1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.acti2(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.acti3(x)

    x = self.conv4(x)

    return x


class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.conv1 = tf.keras.layers.Conv2D(64, (5,5), strides = (2, 2), padding='same', input_shape = (80, 64, 3))

    self.acti1 = tf.keras.layers.LeakyReLU()
    self.drop1 = tf.keras.layers.Dropout(0.2)

    self.conv2 = tf.keras.layers.Conv2D(128, (5,5), strides = (2, 2), padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.acti2 = tf.keras.layers.LeakyReLU()
    self.drop2 = tf.keras.layers.Dropout(0.2)

    self.conv3 = tf.keras.layers.Conv2D(256, (5,5), strides = (1, 1), padding='same')
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.acti3 = tf.keras.layers.LeakyReLU()
    self.drop3 = tf.keras.layers.Dropout(0.2)

    self.flat5 = tf.keras.layers.Flatten()
    self.dense5 = tf.keras.layers.Dense(1)
    self.acti5 = tf.keras.layers.LeakyReLU()

  def call(self, x):
    x = self.conv1(x)
    x = self.acti1(x)
    x = self.drop1(x)
    
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.acti2(x)
    x = self.drop2(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.acti3(x)
    x = self.drop3(x)

    x = self.flat5(x)
    x = self.dense5(x)
    x = self.acti5(x)

    return x

generator = Generator()
discriminator = Discriminator()

print("models OK")

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output)-0.01, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss+fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(0.0002)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002)

@tf.function
def train_step(images, batch_size, bruit, on_generator, on_discriminator):

    if bruit == 'normal' : noise = tf.random.normal([batch_size, noise_dim], 0, 1)
    if bruit == 'uniform' : noise = tf.random.uniform([batch_size, noise_dim], -1, 1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      generated_images = generator(noise)

      real_output = discriminator(images)
      fake_output = discriminator(generated_images)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

      if on_generator :

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

      if on_discriminator :

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss



EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 16
batch_size = 64
nb_train, nb_test = 60000, 10000
ng = 1
nd = 1
verbosity = 100
bruit = 'normal'

train_ds, test_ds = get_data(batch_size, nb_train, nb_test)

if bruit == 'normal' : seed = tf.random.normal([num_examples_to_generate, noise_dim], 0, 1)
if bruit == 'uniform' : seed = tf.random.uniform([num_examples_to_generate, noise_dim], -1, 1)


def train(dataset, epochs):

  now = datetime.datetime.now()
  dir_name = "DCGAN-"+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"-"+str(now.hour)+"_"+str(now.minute)+"-ep:"+str(0).zfill(4)

  if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    os.chdir(dir_name)
  else :
    raise Exception('Dossier existant')

  Lgen_loss = []
  Ldisc_loss = []
  X = []
  j = 0
  last_epoch_time = "<not defined>"

  generate_and_save_images(generator, 0 + 1, seed, training = True)

  for epoch in range(1, epochs+1):
    start = time.time()

    i = 0
    for image_batch in dataset:

        #print("img_batch : "+str(image_batch.shape))

        i += 1
        j += 1

        gen_loss, disc_loss = train_step(image_batch, batch_size, bruit, on_generator=not i%ng, on_discriminator=not i%nd)


        print("\nepoch "+str(epoch)+" - "+str(i)+"/"+str(int(nb_train/batch_size))+" --- "+str(int(i/nd))+" for critic --- "+str(int(i/ng))+" for generator")
        print("gen_loss : "+str(round(gen_loss.numpy(), 3))+" -- disc_loss : "+str(round(disc_loss.numpy(), 3)))
        print ('time : {} sec (last epoch : {} sec)'.format(int(time.time()-start), last_epoch_time))


        if i % verbosity == 0:
          X.append(j)
          Lgen_loss.append(gen_loss)
          Ldisc_loss.append(disc_loss)
          generate_and_save_images(generator, epoch, seed, training = True)


    last_epoch_time = int(time.time()-start)

    generate_and_save_images(generator, epoch, seed)

    fig = plt.figure(figsize=(4,4))

    plt.plot(X,Lgen_loss, label = 'gen_loss')
    plt.plot(X,Ldisc_loss, label = 'disc_loss')
    plt.legend()
    
    plt.savefig('DCGAN_loss')

    plt.clf()
    plt.cla()
    plt.close()

    #saving
    model_path = "tf_saved"
    tf.saved_model.save(generator, model_path)

    os.chdir('..')
    new_dir_name = dir_name[:-4]+str(epoch).zfill(4)
    os.rename(dir_name, new_dir_name)
    os.chdir(new_dir_name)
    dir_name = new_dir_name


def generate_and_save_images(model, epoch, test_input, training = False):
  predictions = model(test_input)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, :]+1)/2)
      plt.axis('off')

  if training :
    plt.savefig('last_one.png')
  else:
    plt.savefig('DCGAN_image_at_epoch_{:04d}.png'.format(epoch))
  plt.clf()
  plt.cla()
  plt.close()

train(train_ds, EPOCHS)