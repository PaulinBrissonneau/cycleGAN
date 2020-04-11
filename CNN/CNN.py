from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf. enable_eager_execution()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train, x_test = x_train.reshape(60000, 28, 28) / 255.0, x_test.reshape(10000, 28, 28) / 255.0

# nécessaire pour créer un objet Dataset
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



class CNN(tf.keras.Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu')
    self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(128, activation='relu')
    self.d2 = tf.keras.layers.Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = CNN()

print("OK")

loss_fct = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#une itération d'entrainement (un batch)
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fct(labels, predictions)

        print(predictions.shape)
        print(labels.shape)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)



EPOCHS = 5

print("OK2")
for epoch in range(EPOCHS):

    i = 0
    for images, labels in train_ds:
        i += 1
        if i%40 == 0 :
            print(str(i)+'/'+str(60000/32)+' batchs')
        train_step(images, labels)

    """
    #pour tracer la courbe de test
    i = 0
    for test_images, test_labels in test_ds:
        i += 1
        if i%40 == 0 :
            print(str(i)+'/'+str(10000/32)+' batchs')
        test_step(test_images, test_labels)"""

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

""" # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()"""
