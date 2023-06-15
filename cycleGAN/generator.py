import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

def Generator(in_channels, out_channels, residual_blocks):
    # Initialise weights
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Define input shape
    inputs = tf.keras.Input(shape=(256, 256, in_channels))
    
    # Encoder
    x = layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same')(inputs)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual Blocks
    for _ in range(residual_blocks):
        residual = x
        x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(out_channels, 
                              (7, 7), 
                              strides=(1, 1), 
                              padding='same', 
                              kernel_initializer=initializer,
                              activation='tanh')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(generated):
  loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)  
  
  return loss_obj(tf.ones_like(generated), generated)


if __name__ == "__main__":
  pass