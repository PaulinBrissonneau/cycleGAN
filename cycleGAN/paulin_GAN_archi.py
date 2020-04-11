import tensorflow as tf
import numpy as np

#168x216

class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()

    self.conv1 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.acti1 = tf.keras.layers.ReLU()
    
    self.conv2 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.acti2 = tf.keras.layers.ReLU()

    self.conv21 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.bn21 = tf.keras.layers.BatchNormalization()
    self.acti21 = tf.keras.layers.ReLU()

    self.conv3 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.acti3 = tf.keras.layers.ReLU()

    #self.conv31 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    #self.bn31 = tf.keras.layers.BatchNormalization()
    #self.acti31 = tf.keras.layers.ReLU()

    self.conv4 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.bn4 = tf.keras.layers.BatchNormalization()
    self.acti4 = tf.keras.layers.ReLU()

    self.conv5 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')

  @tf.function
  def call(self, x):
 

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.acti1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.acti2(x)

    x = self.conv21(x)
    x = self.bn21(x)
    x = self.acti21(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.acti3(x)

    #x = self.conv31(x)
    #x = self.bn31(x)
    #x = self.acti31(x)

    x = self.conv4(x)
    x = self.bn4(x)
    x = self.acti4(x)

    x = self.conv5(x)

    return x


class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.conv1 = tf.keras.layers.Conv2D(64, (5,5), strides = (2, 2), padding='same')

    self.acti1 = tf.keras.layers.LeakyReLU()
    self.drop1 = tf.keras.layers.Dropout(0.2)

    self.conv2 = tf.keras.layers.Conv2D(256, (5,5), strides = (2, 2), padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.acti2 = tf.keras.layers.LeakyReLU()
    self.drop2 = tf.keras.layers.Dropout(0.2)

    self.conv21 = tf.keras.layers.Conv2D(512, (5,5), strides = (2, 2), padding='same')
    self.bn21 = tf.keras.layers.BatchNormalization()
    self.acti21 = tf.keras.layers.LeakyReLU()
    self.drop21 = tf.keras.layers.Dropout(0.2)

    #self.conv3 = tf.keras.layers.Conv2D(1024, (5,5), strides = (2, 2), padding='same')
    #self.bn3 = tf.keras.layers.BatchNormalization()
    #self.acti3 = tf.keras.layers.LeakyReLU()
    #self.drop3 = tf.keras.layers.Dropout(0.2)

    self.conv31 = tf.keras.layers.Conv2D(1024, (5,5), strides = (2, 2), padding='same')
    self.bn31 = tf.keras.layers.BatchNormalization()
    self.acti31 = tf.keras.layers.LeakyReLU()
    self.drop31 = tf.keras.layers.Dropout(0.2)

    self.flat5 = tf.keras.layers.Flatten()
 
    self.dense5 = tf.keras.layers.Dense(1)
    self.acti5 = tf.keras.layers.LeakyReLU()

  @tf.function
  def call(self, x):
    x = self.conv1(x)
    x = self.acti1(x)
    x = self.drop1(x)
    
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.acti2(x)
    x = self.drop2(x)

    x = self.conv21(x)
    x = self.bn21(x)
    x = self.acti21(x)
    x = self.drop21(x)

    #x = self.conv3(x)
    #x = self.bn3(x)
    #x = self.acti3(x)
    #x = self.drop3(x)

    x = self.conv31(x)
    x = self.bn31(x)
    x = self.acti31(x)
    x = self.drop31(x)

    x = self.flat5(x)
    x = self.dense5(x)
    x = self.acti5(x)

    return x


def create_generator(_input_shape = (None, 80, 64, 3)):

    generator = Generator()
    generator.build(input_shape=_input_shape)

    print(generator.summary())

    generator_optimizer = tf.keras.optimizers.Adam(0.0002)


    print("model generator OK")

    return {'model' : generator, 'opti' : generator_optimizer}
   




def create_discriminator(_input_shape = (None, 80, 64, 3)):


    discriminator = Discriminator()
    discriminator.build(input_shape = _input_shape)

    print(discriminator.summary())

    discriminator_optimizer = tf.keras.optimizers.Adam(0.0002)

    print("model discriminator OK")

    return {'model' : discriminator, 'opti' : discriminator_optimizer}


