import os
import sys
import csv
import time
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from IPython import display
from generator import generator_loss
from discriminator import discriminator_loss
from parameters import CHECKPOINT_DIR, RESULTS_DIR
from metrics import shannon_entropy, calculate_metrics
from utils import generate_images, set_checkpoints, initialise_model, initialise_optimiser, tensorboard_init, load_dataset, prepare_data

@tf.function
def train_step(input_image, target, gen, disc, gen_opt, disc_opt):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = gen(input_image, training=True)

        disc_real_output = disc([input_image, target], training=True)
        disc_generated_output = disc([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            gen.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                disc.trainable_variables)

    gen_opt.apply_gradients(zip(generator_gradients,
                                            gen.trainable_variables))
    disc_opt.apply_gradients(zip(discriminator_gradients,
                                                disc.trainable_variables))

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss      

def prepare_train_dataset(train_ds, code, shuffle, folder_name):
    
    # Initialise dataset storage variables
    train_ds_input = []
    train_ds_target = []

    # Prepare dataset according to inputted code
    tf.print("Preparing training dataset...\n", output_stream=os.path.join('file://' + folder_name, "log.out"))
    count = 1
    for image_x, image_y in tf.data.Dataset.zip((train_ds[0], train_ds[1])):
        shann = shannon_entropy(image_y[:,:,:,0])
        if shann > 0.75:
            if count % 12 == 0:
                image_x, image_y = prepare_data(code, image_x, image_y)
                train_ds_input.append(tf.image.rot90(tf.squeeze(image_x, axis=0)))
                train_ds_target.append(tf.image.rot90(tf.squeeze(image_y, axis=0)))   
        if count % 1000 == 0: 
            tf.print(f"Processed: [{((count)/(len(train_ds[0])))*100:.2f}%].", output_stream=os.path.join('file://' + folder_name, "log.out"))
        if count == len(train_ds[0]):
            train_ds_length = len(train_ds_input)            
            train_ds_input = tf.data.Dataset.from_tensor_slices(train_ds_input).batch(1)
            train_ds_target = tf.data.Dataset.from_tensor_slices(train_ds_target).batch(1)
            
            # If shuffle is set to true, will shuffle dataset
            if shuffle:
                train_ds_input = train_ds_input.shuffle(train_ds_length, seed=1)
                train_ds_target = train_ds_target.shuffle(train_ds_length, seed=1)

            tf.print(f"Training dataset prepared of length {train_ds_length}!\n", output_stream=os.path.join('file://' + folder_name, "log.out"))
            break
        count += 1

    # Save training data
    path_train_input = os.path.join(os.getcwd(), os.path.join(code + "_0.75", "train_dataset_input"))
    path_train_target = os.path.join(os.getcwd(), os.path.join(code + "_0.75", "train_dataset_target"))
    train_ds_input.save(path_train_input)
    train_ds_target.save(path_train_target)

    return (train_ds_input, train_ds_target), train_ds_length

def prepare_test_dataset(test_ds, code, shuffle, folder_name):
    
    # Initialise dataset storage variables
    test_ds_input = []
    test_ds_target = []

    # Prepare dataset according to inputted code
    tf.print("Preparing test dataset...\n", output_stream=os.path.join('file://' + folder_name, "log.out"))
    count = 1
    for image_x, image_y in tf.data.Dataset.zip((test_ds[0], test_ds[1])):
        shann = shannon_entropy(image_y[:,:,:,0])
        if shann > 0.75:
            if count % 12 == 0:
                image_x, image_y = prepare_data(code, image_x, image_y)
                test_ds_input.append(tf.image.rot90(tf.squeeze(image_x, axis=0)))
                test_ds_target.append(tf.image.rot90(tf.squeeze(image_y, axis=0)))     
        if count % 1000 == 0: 
            tf.print(f"Processed: [{((count)/(len(test_ds[0])))*100:.2f}%].", output_stream=os.path.join('file://' + folder_name, "log.out"))
        if count == len(test_ds[0]):
            test_ds_length = len(test_ds_input)            
            test_ds_input = tf.data.Dataset.from_tensor_slices(test_ds_input).batch(1)
            test_ds_target = tf.data.Dataset.from_tensor_slices(test_ds_target).batch(1)

            tf.print(f"Test dataset prepared of length {test_ds_length}!\n", output_stream=os.path.join('file://' + folder_name, "log.out"))
        count += 1

    # Save test data
    path_test_input = os.path.join(os.getcwd(), os.path.join(code + "_0.75", "test_dataset_input"))
    path_test_target = os.path.join(os.getcwd(), os.path.join(code + "_0.75", "test_dataset_target"))
    test_ds_input.save(path_test_input)
    test_ds_target.save(path_test_target)

    return (test_ds_input, test_ds_target), test_ds_length

def fit(lr, train_ds, test_ds, epochs, epochs_decay, checkpoint_dir, code, shuffle):

    # Create folder to store images
    folder_name = os.path.join(os.path.join(RESULTS_DIR, code), datetime.now().strftime("%m%d%Y_%H:%M:%S"))
    os.mkdir(folder_name)

    # Set up csv file for metric storage
    filename = os.path.join(folder_name, code + "metrics.csv")
    fields = ['EPOCH', 'PSNR', 'SSIM', 'RSME', 'MAE']
    rows = []

    # Print opening line of the log.out file
    tf.print(f"Training pix2pix model with {code} dataset. \n", output_stream=os.path.join('file://' + folder_name, "log.out"))
    tf.print(f"Learning rate {lr} | Epochs: {epochs} | Epochs before decaying learning rate {epochs - epochs_decay} | Shuffle: {shuffle}\n", output_stream=os.path.join('file://' + folder_name, "log.out"))
    
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 

        # Initialise model
        gen, disc = initialise_model(code)

        # Initialise discriminator loss
        gen_opt, disc_opt = initialise_optimiser(lr)

        # Initialise checkpoints
        checkpoint, checkpoint_prefix = set_checkpoints(checkpoint_dir, gen, disc, gen_opt, disc_opt)

        # Initialise tensorboard writer
        writer = tensorboard_init()

        # Take example input and target image
        example_input, example_target = next(iter(test_ds[0].take(1))), next(iter(test_ds[1].take(1)))
        example_input, example_target = prepare_data(code, example_input, example_target)

        start_training = time.time()
        tf.print(f"Starting training...\n", output_stream=os.path.join('file://' + folder_name, "log.out"))

        for epoch in range(0, epochs):
            start = time.time()

            # Convert step to Tensor to prevent retracing
            epoch = tf.convert_to_tensor(epoch, dtype=tf.int64)
            count = 1

            for train_image_x, train_image_y in tf.data.Dataset.zip((train_ds[0], train_ds[1])):
                if count % 2 == 0:
                    train_image_x, train_image_y = prepare_data(code, train_image_x, train_image_y)
                    gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(train_image_x, train_image_y, gen, disc, gen_opt, disc_opt)

                    with writer.as_default():
                        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch//1)
                        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch//1)
                        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch//1)
                        tf.summary.scalar('disc_loss', disc_loss, step=epoch//1)

                if count % 22 == 0:
                    tf.print('.', end='', output_stream=os.path.join('file://' + folder_name, "log.out"))

                count += 1

            # Using a consistent image so that the progress of the model
            # is clearly visible.
            if (epoch + 1) % 1 == 0:
                PSNR, SSIM, RSME, MAE = generate_images(gen, example_input, example_target, epoch+1, folder_name, code)
                tf.print(f"\nGenerator loss: {gen_total_loss} | Discriminator loss: {disc_loss} \n", output_stream=os.path.join('file://' + folder_name, "log.out"))
                tf.print(f"Epoch: {epoch.numpy() + 1} | SSIM: {SSIM:.3f} | PSNR: {PSNR:.3f} | RSME: {RSME:.3f} | MAE: {MAE:.3f}", output_stream=os.path.join('file://' + folder_name, "log.out"))
                rows.append([epoch, PSNR, SSIM, RSME, MAE])

            # Save (checkpoint) the model every 5k steps
            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            tf.print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                    time.time()-start), output_stream=os.path.join('file://' + folder_name, "log.out"))

            # Decay learning rate
            if (epochs_decay != 0):
                if (epoch + 1 > epochs_decay):
                    step_size = ((lr - 0)/(epochs - epochs_decay))
                    if (epoch + 1) != epochs:
                        gen_opt.lr = gen_opt.lr - step_size
                        disc_opt.lr = disc_opt.lr - step_size
                        tf.print(f"Learning rate has been adjusted to {gen_opt.lr.numpy()}.", output_stream=os.path.join('file://' + folder_name, "log.out"))

        # Write the data rows 
        csvwriter.writerows(rows) 

        # Save model
        gen.save(os.path.join(folder_name, "gen.h5"))

        tf.print(f"Training completed for {code} dataset after {time.time()-start_training:.2f} seconds.\n", output_stream=os.path.join('file://' + folder_name, "log.out"))  

        # Apply trained models on test dataset to calculate mean metrics
        tf.print(f"Testing model performance on test dataset...", output_stream=os.path.join('file://' + folder_name, "log.out"))
        metrics_names = ['SSIM', 'PSNR', 'RSME', 'MAE']
        metrics = {'SSIM': [], 
                'PSNR': [], 
                'RSME': [], 
                'MAE': [],
                }

        count = 1
        for img_idx, test_image_set in enumerate((tf.data.Dataset.zip((test_ds[0], test_ds[1])))):
            if count % 2 == 0:
                test_image_x, test_image_y = prepare_data(code, test_image_set[0], test_image_set[1])
                fake_image_y = gen(test_image_x)
                calculated_metrics = calculate_metrics(tf.squeeze(fake_image_y, axis=0)[:,:,0], tf.squeeze(test_image_y, axis=0)[:,:,0])
                
                for metric_idx, metric in enumerate(calculated_metrics):
                    metrics[metrics_names[metric_idx]].append(metric)

            count += 1

        for metric in metrics_names:   
            mean_metric = np.mean(metrics[metric])
            std_metric = np.std(metrics[metric])
            tf.print(f"The mean {metric} is: {mean_metric:2f} +- {std_metric:2f}.", output_stream=os.path.join('file://' + folder_name, "log.out"))

if __name__ == "__main__":

    # Retrieve args
    parser = argparse.ArgumentParser(description= "Train pix2pix")
    parser.add_argument("--lr", type=float, metavar="", help="Learning rate")
    parser.add_argument("--epochs", type=int, metavar="", help="Number of epochs")
    parser.add_argument("--epochs_decay", type=int, metavar="", help="Number of epochs before decaying learning rate")
    parser.add_argument("--code", type=str, metavar="", help="Code")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, metavar="", help="Shuffle")
    args = parser.parse_args()

    # Print number of GPUs available
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.\n")

    # Set checkpoint directory
    checkpoint_dir = os.path.join(RESULTS_DIR, args.code)

    # Load dataset
    train_ds, test_ds = load_dataset()

    # Shuffle dataset
    if args.shuffle:
        train_ds[0] = train_ds[0].shuffle(len(train_ds[0]), seed=1)
        train_ds[1] = train_ds[1].shuffle(len(train_ds[1]), seed=1)

    # Train pix2pix
    fit(args.lr, train_ds, test_ds, args.epochs, args.epochs_decay, checkpoint_dir, args.code, args.shuffle)