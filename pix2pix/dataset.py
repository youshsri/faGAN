import os
import cv2
import time
import nibabel as nib
import tensorflow as tf
import numpy as np
from parameters import TRAIN_PATH, TEST_PATH, BUFFER_SIZE_TRAIN, BUFFER_SIZE_TEST, BATCH_SIZE

def dataset_lookup_table_init(train_path, test_path):
    train_patients = sorted(os.listdir(train_path))
    test_patients = sorted(os.listdir(test_path))
    patients_list = train_patients + test_patients
    patients_list.remove('.DS_Store')

    train_dataset_lookup = {}
    test_dataset_lookup = {}

    count = 1
    for idx in range(len(train_patients)):
        if train_patients[idx] == '.DS_Store':
            continue
        train_dataset_lookup[count] = train_patients[idx]
        count += 1

    for idx in range(len(test_patients)):
        if test_patients[idx] == '.DS_Store':
            continue
        test_dataset_lookup[count] = test_patients[idx]
        count += 1

    return train_dataset_lookup, test_dataset_lookup


def load(idx, slice_idx, train_dataset, test_dataset, train_path, test_path):
    # Check if patient is in train or test dataset
    if idx <= 359 and idx > 0:
        patient_code = train_dataset[idx]
        path = os.path.join(train_path, patient_code)
    elif idx > 359 and idx <= 512:
        patient_code = test_dataset[idx]
        path = os.path.join(test_path, patient_code)
    else:
        return Exception('Patient does not exist')
    
    # List files in directory of specific patient
    dir_files = sorted(os.listdir(path))
            
    # Get NIFTI file paths
    T1_file = dir_files[dir_files.index(patient_code +  "_T1.nii")]
    T2_file = dir_files[dir_files.index(patient_code + "_T2.nii")]
    T1GD_file = dir_files[dir_files.index(patient_code + "_T1GD.nii")]
    FLAIR_file = dir_files[dir_files.index(patient_code + "_FLAIR.nii")]
    FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
    
    # Load and retrieve fdata
    T1_data = nib.load(os.path.join(path,T1_file)).get_fdata().astype('float32')
    T2_data = nib.load(os.path.join(path,T2_file)).get_fdata().astype('float32')
    T1GD_data = nib.load(os.path.join(path,T1GD_file)).get_fdata().astype('float32')
    FLAIR_data = nib.load(os.path.join(path,FLAIR_file)).get_fdata().astype('float32')
    FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')
        
    # Set placeholder with padding
    strucMRI = -1*np.ones((256, 256, 4))
    FA = -1*np.ones((256, 256, 1))
    
    # Store images into placeholder variables
    strucMRI[8:248, 8:248, 0] = cv2.normalize(T1_data[:,:,slice_idx], None, -1.0, 1.0, cv2.NORM_MINMAX)
    strucMRI[8:248, 8:248, 1] = cv2.normalize(T2_data[:,:,slice_idx], None, -1.0, 1.0, cv2.NORM_MINMAX)
    strucMRI[8:248, 8:248, 2] = cv2.normalize(T1GD_data[:,:,slice_idx], None, -1.0, 1.0, cv2.NORM_MINMAX)
    strucMRI[8:248, 8:248, 3] = cv2.normalize(FLAIR_data[:,:,slice_idx], None, -1.0, 1.0, cv2.NORM_MINMAX)
    FA[8:248, 8:248, 0] = cv2.normalize(FA_data[:,:,slice_idx], None, -1.0, 1.0, cv2.NORM_MINMAX)
    
    return strucMRI.astype('float32'), FA.astype('float32')


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_img = tf.concat([input_image, real_image], axis=2)
        
    cropped_image = tf.image.random_crop(
        stacked_img, size=[256, 256, 5])
    
    return cropped_image[:,:,0:4], cropped_image[:,:,4][:,:,np.newaxis]


@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(idx, slice_idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path):
  input_image, real_image = load(idx, slice_idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path)
  input_image, real_image = random_jitter(input_image, real_image)
  return input_image, real_image


def load_image_test(idx, slice_idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path):
  input_image, real_image = load(idx, slice_idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path)
  input_image, real_image = resize(input_image, real_image,
                                   256, 256)
  return input_image, real_image

def train_dataset_init(train_dataset_lookup, test_dataset_lookup, train_path=TRAIN_PATH, test_path=TEST_PATH):  
    count = 1
    start = time.time()
    print(f"Generating train dataset...\n")

    for idx, _ in train_dataset_lookup.items(): 
        for slice_idx in range(75, 96):
            input_img, target_img = load_image_train(idx, slice_idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path)
            train_dataset = tf.data.Dataset.from_tensor_slices([input_img, target_img])
        print(f"{count}/359.\n")
        count += 1

   # Create train dataset of source images (structural MRI)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE_TRAIN)
    train_dataset = train_dataset.batch(BATCH_SIZE)
          
    print(f"Train dataset generated after {time.time() - start} seconds.\n")
    print(f"Saving train dataset...\n")

    # Save train data
    path_train_dataset = os.path.join(os.getcwd(), "train_dataset")
    train_dataset.save(path_train_dataset)
    
    print(f"Train data saved!\n")
    
    return test_dataset

def test_dataset_init(train_dataset_lookup, test_dataset_lookup, train_path=TRAIN_PATH, test_path=TEST_PATH):  
    count = 1
    start = time.time()
    print(f"Generating test dataset...\n")

    for idx, _ in test_dataset_lookup.items(): 
        for slice_idx in range(75, 96):
            input_img, target_img = load_image_test(idx, slice_idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path)
            test_dataset = tf.data.Dataset.from_tensor_slices([input_img, target_img])
        print(f"{count}/153.\n")
        count += 1

   # Create train dataset of source images (structural MRI)
    test_dataset = test_dataset.shuffle(BUFFER_SIZE_TEST)
    test_dataset = test_dataset.batch(BATCH_SIZE)
          
    print(f"Test dataset generated after {time.time() - start} seconds.\n")
    print(f"Saving test dataset...\n")

    # Save test data
    path_test_dataset = os.path.join(os.getcwd(), "test_dataset")
    test_dataset.save(path_test_dataset)
    
    print(f"Test data saved!\n")
    
    return test_dataset

if __name__ == "__main__":
    # Generate lookup table
    train_lookup, test_lookup = dataset_lookup_table_init(TRAIN_PATH, TEST_PATH)

    # Generate train dataset
    train_dataset = train_dataset_init(train_lookup, test_lookup, train_path=TRAIN_PATH, test_path=TEST_PATH)

    # Generate test dataset
    test_dataset = test_dataset_init(train_lookup, test_lookup, train_path=TRAIN_PATH, test_path=TEST_PATH)
