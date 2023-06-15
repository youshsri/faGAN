import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator
from metrics import calculate_metrics
from parameters import ROOT_DIR_DATASET, TRAIN_INPUT_PATH_DATASET, TRAIN_TARGET_PATH_DATASET, TEST_INPUT_PATH_DATASET, TEST_TARGET_PATH_DATASET, RESULTS_DIR, CODES, CHANNELS, MODALITIES

def tensorboard_init(log_dir= os.path.join(RESULTS_DIR, "logs/")):
    writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    return writer


def prepare_data(code, input, target):
    if code == "T1":        
        return tf.expand_dims(input[:,:,:,0], -1), tf.expand_dims(target[:,:,:,0], -1)
    
    elif code == "T2":
        return tf.expand_dims(input[:,:,:,1], -1), tf.expand_dims(target[:,:,:,0], -1)
    
    elif code == "T1GD":
        return tf.expand_dims(input[:,:,:,2], -1), tf.expand_dims(target[:,:,:,0], -1)
    
    elif code == "FLAIR":
        return tf.expand_dims(input[:,:,:,3], -1), tf.expand_dims(target[:,:,:,0], -1)

    elif code == "T1-T2":
        return tf.stack([input[:,:,:,0],input[:,:,:,1]], axis=3), tf.expand_dims(target[:,:,:,0], -1)
    
    elif code == "T1-T1GD":
        return tf.stack([input[:,:,:,0],input[:,:,:,2]], axis=3), tf.expand_dims(target[:,:,:,0], -1)
    
    elif code == "T1-FLAIR":
         return tf.stack([input[:,:,:,0],input[:,:,:,3]], axis=3), tf.expand_dims(target[:,:,:,0], -1)
    
    elif code == "T2-T1GD":
        return tf.stack([input[:,:,:,1],input[:,:,:,2]], axis=3), tf.expand_dims(target[:,:,:,0], -1)
    
    elif code == "T2-FLAIR":
        return tf.stack([input[:,:,:,1],input[:,:,:,3]], axis=3), tf.expand_dims(target[:,:,:,0], -1)

    elif code == "T1GD-FLAIR":
        return tf.stack([input[:,:,:,2],input[:,:,:,3]], axis=3), tf.expand_dims(target[:,:,:,0], -1)

    elif code == "T1-T2-T1GD":
        return tf.stack([input[:,:,:,0],input[:,:,:,1], input[:,:,:,2]], axis=3), tf.expand_dims(target[:,:,:,0], -1)
    
    elif code == "T1-T2-FLAIR":
        return tf.stack([input[:,:,:,0],input[:,:,:,1], input[:,:,:,3]], axis=3), tf.expand_dims(target[:,:,:,0], -1)

    elif code == "T1-T1GD-FLAIR":
        return tf.stack([input[:,:,:,0],input[:,:,:,2], input[:,:,:,3]], axis=3), tf.expand_dims(target[:,:,:,0], -1)

    elif code == "T2-T1GD-FLAIR":
        return tf.stack([input[:,:,:,1],input[:,:,:,2], input[:,:,:,3]], axis=3), tf.expand_dims(target[:,:,:,0], -1)

    elif code == "all":
        return input, tf.expand_dims(target[:,:,:,0], -1)


def load_dataset( 
                 train_input=TRAIN_INPUT_PATH_DATASET, 
                 train_target=TRAIN_TARGET_PATH_DATASET,
                 test_input=TEST_INPUT_PATH_DATASET,
                 test_target=TEST_TARGET_PATH_DATASET,
                 ):
    train_input_dataset = tf.data.Dataset.load(os.path.join(ROOT_DIR_DATASET, os.path.join(CODES[-1] + "_0.75", train_input)))
    train_target_dataset = tf.data.Dataset.load(os.path.join(ROOT_DIR_DATASET, os.path.join(CODES[-1] + "_0.75", train_target)))
    test_input_dataset = tf.data.Dataset.load(os.path.join(ROOT_DIR_DATASET, os.path.join(CODES[-1] + "_0.75", test_input)))
    test_target_dataset = tf.data.Dataset.load(os.path.join(ROOT_DIR_DATASET, os.path.join(CODES[-1] + "_0.75", test_target)))

    return [train_input_dataset, train_target_dataset], [test_input_dataset, test_target_dataset]


def set_checkpoints(checkpoint_dir, generator, discriminator, generator_optimizer, discriminator_optimizer):
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    return checkpoint, checkpoint_prefix


def generate_images(model, test_input, tar, step, folder_name, code):
    
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    # Get respective channels for the code
    channels = CHANNELS[CODES.index(code)]
    modalities = MODALITIES[CODES.index(code)]

    display_list = [test_input, tar, prediction]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    # Calculate metrics between target image and predicted image
    SSIM, PSNR, RSME, MAE = calculate_metrics(tf.squeeze(display_list[1], axis=0)[:,:,0], 
                                            tf.squeeze(display_list[2], axis=0)[:,:,0],
                                            )

    for i in range(0, channels+2):
        if i < channels:
            plt.subplot(3, 2, i+1)
            plt.title(title[0] + f"({modalities[i]})")
            plt.imshow(tf.squeeze(display_list[0], axis=0)[:,:,i]* 0.5 + 0.5, cmap='gray')
            plt.axis('off')

        elif i == channels:
            plt.subplot(3, 2, 5)
            plt.title(title[1])
            plt.imshow(tf.squeeze(display_list[1], axis=0)[:,:,0]* 0.5 + 0.5, cmap='gray')
            plt.axis('off')

        elif i == channels + 1:
            plt.subplot(3, 2, 6)
            plt.title(title[2])
            plt.imshow(tf.squeeze(display_list[2], axis=0)[:,:,0]* 0.5 + 0.5, cmap='gray')
            plt.axis('off')

    plt.savefig(os.path.join(folder_name, str(step) + ".png"), orientation="landscape") 
    plt.close()

    return PSNR, SSIM, RSME, MAE


def initialise_model(code):
    # Get respective channels for the code
    channels = CHANNELS[CODES.index(code)]

    # Initialise generator and discriminator
    generator = Generator(channels)
    discriminator = Discriminator(channels)

    return generator, discriminator
                       

def initialise_optimiser(lr):
    # Define optimisation parameters
    beta = 0.5

    # Initialise optimisers for generator & discriminator
    generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta)

    return generator_optimizer, discriminator_optimizer


if __name__ == "__main__":
  pass