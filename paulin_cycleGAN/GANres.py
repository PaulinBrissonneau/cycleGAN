import tensorflow as tf
import tensorflow_addons as tf_add
import numpy as np



class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()

    kernel_init = tf.keras.initializers.RandomNormal(stddev=0.02)

    self.conv1 = tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init)
    self.acti1 = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.conv2 = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init)
    self.norm2 = tf_add.layers.InstanceNormalization(axis=-1)
    self.acti2 = tf.keras.layers.LeakyReLU(alpha=0.2)
    #self.drop2 = tf.keras.layers.Dropout(0.2)
    
    self.conv3 = tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init)
    self.norm3 = tf_add.layers.InstanceNormalization(axis=-1)
    self.acti3 = tf.keras.layers.LeakyReLU(alpha=0.2)
    #self.drop3 = tf.keras.layers.Dropout(0.2)
    
    self.conv4 = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init)
    self.norm4 = tf_add.layers.InstanceNormalization(axis=-1)
    self.acti4 = tf.keras.layers.LeakyReLU(alpha=0.2)
    #self.drop4 = tf.keras.layers.Dropout(0.2)
    
    
    self.conv5 = tf.keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=kernel_init)
    self.norm5 = tf_add.layers.InstanceNormalization(axis=-1)
    self.acti5 = tf.keras.layers.LeakyReLU(alpha=0.2)
    
    self.out = tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=kernel_init)


  @tf.function
  def call(self, x):
 
    x = self.conv1(x)
    x = self.acti1(x)

    x = self.conv2(x)
    x = self.norm2(x)
    x = self.acti2(x)
    #x = self.drop2(x)

    x = self.conv3(x)
    x = self.norm3(x)
    x = self.acti3(x)
    #x = self.drop3(x)

    x = self.conv4(x)
    x = self.norm4(x)
    x = self.acti4(x)
    #x = self.drop4(x)

    
    x = self.conv5(x)
    x = self.norm5(x)
    x = self.acti5(x)

    x = self.out(x)

    return x


class ResidualBlock(tf.keras.Model):
    def __init__(self, n_filters):
        super(ResidualBlock, self).__init__()

        kernel_init = tf.keras.initializers.RandomNormal(stddev=0.02)
        
        self.res_conv1 = tf.keras.layers.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=kernel_init)
        self.res_norm1 = tf_add.layers.InstanceNormalization(axis=-1)
        self.res_acti1 = tf.keras.layers.Activation('relu')
        
        self.res_conv2 = tf.keras.layers.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=kernel_init)
        self.res_norm2 = tf_add.layers.InstanceNormalization(axis=-1)

        #self.res_actiout = tf.keras.layers.Activation('relu')
    

    @tf.function
    def call(self, x):

        x_input = x

        x = self.res_conv1(x)
        x = self.res_norm1(x)
        x = self.res_acti1(x)

        x = self.res_conv2(x)
        x = self.res_norm2(x)

        x = tf.add(x_input, x)

        #x = self.res_actiout(x)

        return x


class Generator(tf.keras.Model):
  def __init__(self, n_resnet):
    super(Generator, self).__init__()

    kernel_init = tf.keras.initializers.RandomNormal(stddev=0.02)

    self.conv1 = tf.keras.layers.Conv2D(64, (7,7), padding='same', kernel_initializer=kernel_init)
    self.norm1 = tf_add.layers.InstanceNormalization(axis=-1)
    self.acti1 = tf.keras.layers.Activation('relu')

    self.conv2 = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=kernel_init)
    self.norm2 = tf_add.layers.InstanceNormalization(axis=-1)
    self.acti2 = tf.keras.layers.Activation('relu')


    self.conv3 = tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=kernel_init)
    self.norm3 = tf_add.layers.InstanceNormalization(axis=-1)
    self.acti3 = tf.keras.layers.Activation('relu')

    self.residuals = tf.keras.Sequential()
    for i in range(n_resnet):
        self.residuals.add(ResidualBlock(n_filters=256))

    
    self.convT1 = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=kernel_init)
    self.normT1 = tf_add.layers.InstanceNormalization(axis=-1)
    self.actiT1 = tf.keras.layers.Activation('relu')

    self.convT2 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=kernel_init)
    self.normT2 = tf_add.layers.InstanceNormalization(axis=-1)
    self.actiT2 = tf.keras.layers.Activation('relu')

    self.convT3 = tf.keras.layers.Conv2D(3, (7,7), padding='same', kernel_initializer=kernel_init)
    self.normT3 = tf_add.layers.InstanceNormalization(axis=-1)

    self.out = tf.keras.layers.Activation('tanh')

  @tf.function
  def call(self, x):
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.acti1(x)
    
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.acti2(x)

    x = self.conv3(x)
    x = self.norm3(x)
    x = self.acti3(x)

    x = self.residuals(x)

    x = self.convT1(x)
    x = self.normT1(x)
    x = self.actiT1(x)

    x = self.convT2(x)
    x = self.normT2(x)
    x = self.actiT2(x)

    x = self.convT3(x)
    x = self.normT3(x)

    x = self.out(x)

    return x


def create_generator(_n_resnet, _input_shape = (None, 80, 64, 3)):

    generator = Generator(_n_resnet)
    generator.build(input_shape=_input_shape)

    print(generator.summary())

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)


    print("model generator OK")

    return {'model' : generator, 'opti' : generator_optimizer}
   




def create_discriminator(_input_shape = (None, 80, 64, 3)):


    discriminator = Discriminator()
    discriminator.build(input_shape = _input_shape)

    print(discriminator.summary())

    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    print("model discriminator OK")

    return {'model' : discriminator, 'opti' : discriminator_optimizer}


