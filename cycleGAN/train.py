import os
import csv
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from metrics import calculate_metrics, shannon_entropy
from utils import generate_images
from loss import calc_cycle_loss, identity_loss
from generator import generator_loss
from discriminator import discriminator_loss
from utils import set_checkpoints, initialise_models, initialise_optimisers, tensorboard_init, load_dataset, prepare_data
from parameters import RESULTS_DIR, CODES, CHANNELS

@tf.function
def train_step(real_x, real_y, gen_g, gen_f, disc_x, disc_y, gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = gen_g(real_x, training=True)
        cycled_x = gen_f(fake_y, training=True)

        fake_x = gen_f(real_y, training=True)
        cycled_y = gen_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = gen_f(real_x, training=True)
        same_y = gen_g(real_y, training=True)

        disc_real_x = disc_x(real_x, training=True)
        disc_real_y = disc_y(real_y, training=True)

        disc_fake_x = disc_x(fake_x, training=True)
        disc_fake_y = disc_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                            gen_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                            gen_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                disc_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                disc_y.trainable_variables)

    # Apply the gradients to the optimizer
    gen_g_opt.apply_gradients(zip(generator_g_gradients, 
                                                gen_g.trainable_variables))

    gen_f_opt.apply_gradients(zip(generator_f_gradients, 
                                                gen_f.trainable_variables))

    disc_x_opt.apply_gradients(zip(discriminator_x_gradients,
                                                    disc_x.trainable_variables))

    disc_y_opt.apply_gradients(zip(discriminator_y_gradients,
                                                    disc_y.trainable_variables))
    
    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss   


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
            # train_ds_input = tf.data.Dataset.from_tensor_slices(train_ds_input).batch(1)
            # print(next(iter(train_ds_input.take(1))))
            # len(train_ds[0])
        if count == len(train_ds[0]):
            train_ds_length = len(train_ds_input)            
            train_ds_input = tf.data.Dataset.from_tensor_slices(train_ds_input).batch(1)
            train_ds_target = tf.data.Dataset.from_tensor_slices(train_ds_target).batch(1)
            
            # If shuffle is set to true, will shuffle dataset
            if shuffle:
                train_ds_input = train_ds_input.shuffle(train_ds_length, seed=1)
                train_ds_target = train_ds_target.shuffle(train_ds_length, seed=2)

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


def fit(lr, train_ds, test_ds, steps, steps_decay, checkpoint_dir, code, shuffle, residual_blocks):

    # Create folder to store images
    folder_name = os.path.join(os.path.join(RESULTS_DIR, code), datetime.now().strftime("%m%d%Y_%H:%M:%S"))
    os.mkdir(folder_name)

    # Set up csv file for metric storage
    filename = os.path.join(folder_name, code + "metrics.csv")
    fields = ['EPOCH', 'PSNR', 'SSIM', 'RSME', 'MAE']
    rows = []

    # Print opening line of the log.out file
    tf.print(f"Training cycleGAN model with {code} dataset. \n", output_stream=os.path.join('file://' + folder_name, "log.out"))
    tf.print(f"Learning rate {lr} | Epochs: {steps} | Epochs with linear learning rate decay: {steps_decay} | Residual blocks: {9} | Shuffle: {shuffle}\n", output_stream=os.path.join('file://' + folder_name, "log.out"))
    
    train_ds_length = round(len(train_ds[0])/2)
    test_ds_length = round(len(test_ds[0])/2)

    # Writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 

        # Initialise models and optimisers
        gen_g, gen_f, disc_x, disc_y = initialise_models(code, residual_blocks)
        gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt = initialise_optimisers(lr, steps, steps_decay, train_ds_length)

        # Initialise checkpoints
        ckpt, ckpt_manager, checkpoint_prefix = set_checkpoints(checkpoint_dir, 
                                                                gen_g, 
                                                                gen_f, 
                                                                disc_x, 
                                                                disc_y, 
                                                                gen_g_opt,
                                                                gen_f_opt,
                                                                disc_x_opt,
                                                                disc_y_opt)

        # Initialise tensorboard writer
        writer = tensorboard_init()

        # Take example input and target image
        example_input, example_target = next(iter(test_ds[0].take(1))), next(iter(test_ds[1].take(1)))

        start_training = time.time()
        tf.print(f"Starting training...\n", output_stream=os.path.join('file://' + folder_name, "log.out"))

        for epoch in range(steps):
            start = time.time()

            # Convert step to Tensor to prevent retracing
            epoch = tf.convert_to_tensor(epoch, dtype=tf.int64)
            count = 1

            for train_image_x, train_image_y in tf.data.Dataset.zip((train_ds[0], train_ds[1])):
                if count % 2 == 0:
                    train_image_x, train_image_y = prepare_data(code, train_image_x, train_image_y)
                    total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss = train_step(train_image_x, train_image_y, gen_g, gen_f, disc_x, disc_y, gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt)

                    with writer.as_default():
                        tf.summary.scalar('total_gen_g_loss', total_gen_g_loss, step=epoch//1)
                        tf.summary.scalar('total_gen_f_loss', total_gen_f_loss, step=epoch//1)
                        tf.summary.scalar('disc_x_loss', disc_x_loss, step=epoch//1)
                        tf.summary.scalar('disc_y_loss', disc_y_loss, step=epoch//1)

                if count % 22 == 0:
                    tf.print('.', end='', output_stream=os.path.join('file://' + folder_name, "log.out"))

                count += 1 

            # Using a consistent image so that the progress of the model
            # is clearly visible.
            if (epoch + 1) % 1 == 0:
                PSNR, SSIM, RSME, MAE = generate_images(gen_g, example_input, example_target, epoch+1, folder_name, code)
                tf.print(f"\nEpoch: {epoch.numpy() + 1} | SSIM: {SSIM:.3f} | PSNR: {PSNR:.3f} | RSME: {RSME:.3f} | MAE: {MAE:.3f}", output_stream=os.path.join('file://' + folder_name, "log.out"))
                rows.append([epoch, PSNR, SSIM, RSME, MAE])

            # Decay learning rate
            if (steps_decay != 0):
                if (epoch + 1 > steps_decay):
                    step_size = ((lr - 0)/(steps - steps_decay))
                    if (epoch + 1) != steps:
                        gen_g_opt.lr = gen_g_opt.lr - step_size
                        gen_f_opt.lr = gen_f_opt.lr - step_size
                        disc_x_opt.lr = disc_x_opt.lr - step_size
                        disc_y_opt.lr = disc_y_opt.lr - step_size
                        tf.print(f"Learning rate has been adjusted to {gen_g_opt.lr.numpy()}.", output_stream=os.path.join('file://' + folder_name, "log.out"))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                tf.print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                        ckpt_save_path), output_stream=os.path.join('file://' + folder_name, "log.out"))
            
            tf.print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                    time.time()-start), output_stream=os.path.join('file://' + folder_name, "log.out"))

        # Write the data rows 
        csvwriter.writerows(rows) 

        # Save model
        gen_g.save(os.path.join(folder_name, "gen_g.h5"))
        gen_f.save(os.path.join(folder_name, "gen_f.h5")) 

        tf.print(f"Training completed for {code} dataset after {time.time()-start_training:.2f} seconds.\n", output_stream=os.path.join('file://' + folder_name, "log.out"))  

        # Apply trained models on test dataset to calculate mean metrics
        tf.print(f"Testing model performance on test dataset...", output_stream=os.path.join('file://' + folder_name, "log.out"))
        metrics_names = ['SSIM', 'PSNR', 'RSME', 'MAE']
        metrics = {'SSIM': 0, 
                'PSNR': 0, 
                'RSME': 0, 
                'MAE': 0}

        count = 1
        for img_idx, test_image_set in enumerate((tf.data.Dataset.zip((test_ds[0], test_ds[1])))):
            if count % 2 == 0:
                fake_image_y = gen_g(test_image_set[0])
                calculated_metrics = calculate_metrics(tf.squeeze(fake_image_y, axis=0)[:,:,0], tf.squeeze(test_image_set[1], axis=0)[:,:,0])
            
                for metric_idx, metric in enumerate(calculated_metrics):
                    metrics[metrics_names[metric_idx]] += metric
            
            count += 1

        for metric in metrics_names:   
            mean_metric = metrics[metric]/test_ds_length
            tf.print(f"The mean {metric} is: {mean_metric:2f}.", output_stream=os.path.join('file://' + folder_name, "log.out"))


if __name__ == "__main__":

    # Retrieve args
    parser = argparse.ArgumentParser(description= "Train cycleGAN")
    parser.add_argument("--lr", type=float, metavar="", help="Learning rate")
    parser.add_argument("--residual_blocks", type=int, metavar="", help="Number of residual blocks")
    parser.add_argument("--epochs", type=int, metavar="", help="Number of epochs")
    parser.add_argument("--epochs_decay", type=int, metavar="", help="Number of epochs before decaying learning rate")
    parser.add_argument("--code", type=str, metavar="", help="Code")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, metavar="", help="Shuffle")
    parser.add_argument("--wasserstein_loss", action=argparse.BooleanOptionalAction, metavar="", help="Wasserstein Loss")
    args = parser.parse_args()

    # Print number of GPUs available
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.\n")

    # Set checkpoint directory
    checkpoint_dir = os.path.join(RESULTS_DIR, args.code)

    # Generate dataset
    train_ds, test_ds = load_dataset()
    train_ds[1] = train_ds[1].shuffle(len(train_ds[1]), seed=1)

    # Shuffle dataset
    if args.shuffle:
        train_ds[0] = train_ds[0].shuffle(len(train_ds[0]), seed=2)

    # Train cycleGAN
    fit(args.lr, train_ds, test_ds, args.epochs, args.epochs_decay, checkpoint_dir, args.code, args.shuffle, args.residual_blocks)