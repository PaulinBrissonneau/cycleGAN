from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
import numpy as np

class cycleGAN(Model):
    """
    Honestly, the best way to implement a model
    """

    def __init__(self, **kwargs):
        """
        initialize model with its super's features and methods
        """

        super(cycleGAN, self).__init__()
        self.__dict__.update(kwargs)

    @tf.function
    def A_to_B(self, x):
        """
        passes through the generator network
        """

        return self.gen_A_to_B(x)

    @tf.function
    def B_to_A(self, x):
        """
        passes through the generator network
        """

        return self.gen_B_to_A(x)

    @tf.function
    def discriminate_A(self, x):
        """
        passes through the discrimenator network
        """

        return self.disc_A(x)

    @tf.function
    def discriminate_B(self, x):
        """
        passes through the discrimenator network
        """

        return self.disc_B(x)

    @tf.function
    def compute_A_to_B_loss(self, real_a, real_b, batch_size=1):
        """
        passes through the network and computes loss
        """

        # forward cycle
        gen_b = self.A_to_B(real_a)
        forward_output = self.B_to_A(gen_b)
        # discriminator element
        discriminator_output = self.discriminate_B(gen_b)
        # backward cycle
        gen_a = self.B_to_A(real_b)
        backward_output = self.A_to_B(gen_a)
        # identity element
        identity_output = self.A_to_B(real_b)
        # calculate different losses
        discrimenator_loss = tf.reduce_mean(tf.keras.losses.MSE(
            discriminator_output, tf.ones(shape=discriminator_output.shape)))
        identity_loss = tf.reduce_mean(
            tf.keras.losses.MAE(identity_output, real_b))
        forward_loss = tf.reduce_mean(
            tf.keras.losses.MAE(forward_output,  real_a))
        backward_loss = tf.reduce_mean(
            tf.keras.losses.MAE(backward_output, real_b))
        # calculate final loss
        loss = 1.*discrimenator_loss + 5.*identity_loss + \
            10.*forward_loss + 10.*backward_loss

        return loss

    @tf.function
    def compute_B_to_A_loss(self, real_a, real_b, batch_size=1):
        """
        passes through the network and computes loss
        """

        # forward cycle
        gen_a = self.B_to_A(real_b)
        forward_output = self.A_to_B(gen_a)
        # discriminator element
        discriminator_output = self.discriminate_A(gen_a)
        # backward cycle
        gen_b = self.A_to_B(real_a)
        backward_output = self.B_to_A(gen_b)
        # identity element
        identity_output = self.B_to_A(real_a)
        # calculate different losses
        discrimenator_loss = tf.reduce_mean(tf.keras.losses.MSE(
            discriminator_output, tf.ones(shape=discriminator_output.shape)))
        identity_loss = tf.reduce_mean(
            tf.keras.losses.MAE(identity_output, real_a))
        forward_loss = tf.reduce_mean(
            tf.keras.losses.MAE(forward_output,  real_b))
        backward_loss = tf.reduce_mean(
            tf.keras.losses.MAE(backward_output, real_a))
        # calculate final loss
        loss = 1.*discrimenator_loss + 5.*identity_loss + \
            10.*forward_loss + 10.*backward_loss

        return loss

    @tf.function
    def compute_disc_A_loss(self, real_a, real_b, batch_size=1):
        """
        passes through the network and computes loss
        """

        # generate fake sample
        gen_a = self.B_to_A(real_b)
        gen_a = self.buffer_A.update(gen_a)

        # discriminator element
        real_output = self.discriminate_A(real_a)
        fake_output = self.discriminate_A(gen_a)

        # calculate different losses
        real_loss = tf.reduce_mean(tf.keras.losses.MSE(
            real_output, tf.ones(shape=real_output.shape)))
        fake_loss = tf.reduce_mean(tf.keras.losses.MSE(
            fake_output, tf.zeros(shape=fake_output.shape)))

        # calculate final loss
        loss = .5*(real_loss + fake_loss)

        return loss

    @tf.function
    def compute_disc_B_loss(self, real_a, real_b, batch_size=1):
        """
        passes through the network and computes loss
        """
        # generate fake sample
        gen_b = self.A_to_B(real_a)
        gen_b = self.buffer_B.update(gen_b)

        # discriminator element
        real_output = self.discriminate_B(real_b)
        fake_output = self.discriminate_B(gen_b)

        # calculate different losses
        real_loss = tf.reduce_mean(tf.keras.losses.MSE(
            real_output, tf.ones(shape=real_output.shape)))
        fake_loss = tf.reduce_mean(tf.keras.losses.MSE(
            fake_output, tf.zeros(shape=fake_output.shape)))

        # calculate final loss
        loss = .5*(real_loss + fake_loss)

        return loss

    @tf.function
    def compute_A_to_B_gradients(self, real_a, real_b, batch_size=1):
        """
        passes through the network and computes loss
        """

        # computation envirenment
        with tf.GradientTape() as A_to_B_tape:
            # compute loss
            A_to_B_loss = self.compute_A_to_B_loss(real_a, real_b, batch_size)
            # compute gradients
            A_to_B_gradients = A_to_B_tape.gradient(A_to_B_loss, self.gen_A_to_B.trainable_variables)

        return A_to_B_gradients, A_to_B_loss

    @tf.function
    def compute_B_to_A_gradients(self, real_a, real_b, batch_size=1):
        """
        passes through the network and computes loss
        """

        # computation envirenment
        with tf.GradientTape() as B_to_A_tape:
            # compute loss
            B_to_A_loss = self.compute_B_to_A_loss(real_a, real_b, batch_size)
            # compute gradients
            B_to_A_gradients = B_to_A_tape.gradient(B_to_A_loss, self.gen_B_to_A.trainable_variables)

        return B_to_A_gradients, B_to_A_loss

    @tf.function
    def compute_disc_A_gradients(self, real_a, real_b, batch_size=1):
        """
        passes through the network and computes loss
        """

        # computation envirenment
        with tf.GradientTape() as disc_A_tape:
            # compute loss
            disc_A_loss = self.compute_disc_A_loss(real_a, real_b, batch_size)
            # compute gradients
            disc_A_gradients = disc_A_tape.gradient(disc_A_loss, self.disc_A.trainable_variables)

        return disc_A_gradients, disc_A_loss

    @tf.function
    def compute_disc_B_gradients(self, real_a, real_b, batch_size=1):
        """
        passes through the network and computes loss
        """

        # computation envirenment
        with tf.GradientTape() as disc_B_tape:
            # compute loss
            disc_B_loss = self.compute_disc_B_loss(real_a, real_b, batch_size)
            # compute gradients
            disc_B_gradients = disc_B_tape.gradient(disc_B_loss, self.disc_B.trainable_variables)

        return disc_B_gradients, disc_B_loss

    @tf.function
    def train_A_to_B(self, real_a, real_b, batch_size=1):

        # calculate gradients and loss
        A_to_B_gradients, A_to_B_loss = self.compute_A_to_B_gradients(
            real_a, real_b, batch_size)
        # apply gradients
        self.gen_A_to_B_optimizer.apply_gradients(
            zip(A_to_B_gradients, self.gen_A_to_B.trainable_variables))

        return A_to_B_loss

    @tf.function
    def train_B_to_A(self, real_a, real_b, batch_size=1):

        # calculate gradients and loss
        B_to_A_gradients, B_to_A_loss = self.compute_B_to_A_gradients(real_a, real_b, batch_size)
        # apply gradients
        self.gen_B_to_A_optimizer.apply_gradients(zip(B_to_A_gradients, self.gen_B_to_A.trainable_variables))

        return B_to_A_loss

    @tf.function
    def train_disc_A(self, real_a, real_b, batch_size=1):

        # calculate gradients and loss
        disc_A_gradients, disc_A_loss = self.compute_disc_A_gradients(
            real_a, real_b, batch_size)
        # apply gradients
        self.disc_A_optimizer.apply_gradients(zip(disc_A_gradients, self.disc_A.trainable_variables))

        return disc_A_loss

    @tf.function
    def train_disc_B(self, real_a, real_b, batch_size=1):

        # calculate gradients and loss
        disc_B_gradients, disc_B_loss = self.compute_disc_B_gradients(
            real_a, real_b, batch_size)
        # apply gradients
        self.disc_B_optimizer.apply_gradients(
            zip(disc_B_gradients, self.disc_B.trainable_variables))

        return disc_B_loss