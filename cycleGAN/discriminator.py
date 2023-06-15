import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

def Discriminator(in_channels):
  # Initialise weights
  initializer = tf.random_normal_initializer(0., 0.02)

  # Define input shape
  inputs = tf.keras.Input(shape=(256, 256, in_channels))

  x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inputs)
  x = layers.LeakyReLU(0.2)(x)

  x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = layers.LeakyReLU(0.2)(x)

  x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = layers.LeakyReLU(0.2)(x)

  x = layers.Conv2D(512, (4, 4), strides=(1, 1), padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = layers.LeakyReLU(0.2)(x)

  x = layers.Conv2D(1, (4, 4), strides=(1, 1), kernel_initializer=initializer, padding='same')(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator_loss(real, generated):
  loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

if __name__ == "__main__":
  pass