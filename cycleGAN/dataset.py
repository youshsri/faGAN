import os
import cv2
import time
from metrics import shannon_entropy
import nibabel as nib
import tensorflow as tf
import numpy as np
from parameters import TRAIN_PATH, TEST_PATH, BUFFER_SIZE_TRAIN, BUFFER_SIZE_TEST, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CODES


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


def load(idx, train_dataset, test_dataset, train_path, test_path, code="all"):
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
    
    if code == "T1":
        num_modalities = 1

        T1_file = dir_files[dir_files.index(patient_code +  "_T1.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]  

        T1_data = nib.load(os.path.join(path,T1_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')
        
        return [T1_data], FA_data, num_modalities
    
    elif code == "T2":
        num_modalities = 1

        T2_file = dir_files[dir_files.index(patient_code + "_T2.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]  

        T2_data = nib.load(os.path.join(path,T2_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T2_data], FA_data, num_modalities
    
    elif code == "T1GD":
        num_modalities = 1

        T1GD_file = dir_files[dir_files.index(patient_code + "_T1GD.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]  

        T1GD_data = nib.load(os.path.join(path,T1GD_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T1GD_data], FA_data, num_modalities
    
    elif code == "FLAIR":
        num_modalities = 1

        FLAIR_file = dir_files[dir_files.index(patient_code + "_FLAIR.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]  

        FLAIR_data = nib.load(os.path.join(path,FLAIR_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [FLAIR_data], FA_data, num_modalities

    elif code == "T1-T2":
        num_modalities = 2

        T1_file = dir_files[dir_files.index(patient_code +  "_T1.nii")]
        T2_file = dir_files[dir_files.index(patient_code + "_T2.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T1_data = nib.load(os.path.join(path,T1_file)).get_fdata().astype('float32')
        T2_data = nib.load(os.path.join(path,T2_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T1_data, T2_data], FA_data, num_modalities
    
    elif code == "T1-T1GD":
        num_modalities = 2
        
        T1_file = dir_files[dir_files.index(patient_code +  "_T1.nii")]
        T1GD_file = dir_files[dir_files.index(patient_code + "_T1GD.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T1_data = nib.load(os.path.join(path,T1_file)).get_fdata().astype('float32')
        T1GD_data = nib.load(os.path.join(path,T1GD_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T1_data, T1GD_data], FA_data, num_modalities
    
    elif code == "T1-FLAIR":
        num_modalities = 2
        
        T1_file = dir_files[dir_files.index(patient_code +  "_T1.nii")]
        FLAIR_file = dir_files[dir_files.index(patient_code + "_FLAIR.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T1_data = nib.load(os.path.join(path,T1_file)).get_fdata().astype('float32')
        FLAIR_data = nib.load(os.path.join(path,FLAIR_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T1_data, FLAIR_data], FA_data, num_modalities
    
    elif code == "T2-T1GD":
        num_modalities = 2

        T2_file = dir_files[dir_files.index(patient_code + "_T2.nii")]
        T1GD_file = dir_files[dir_files.index(patient_code + "_T1GD.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T2_data = nib.load(os.path.join(path,T2_file)).get_fdata().astype('float32')
        T1GD_data = nib.load(os.path.join(path,T1GD_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T2_data, T1GD_data], FA_data, num_modalities
    
    elif code == "T2-FLAIR":
        num_modalities = 2
        
        T2_file = dir_files[dir_files.index(patient_code + "_T2.nii")]
        FLAIR_file = dir_files[dir_files.index(patient_code + "_FLAIR.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T2_data = nib.load(os.path.join(path,T2_file)).get_fdata().astype('float32')
        FLAIR_data = nib.load(os.path.join(path,FLAIR_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T2_data, FLAIR_data], FA_data, num_modalities

    elif code == "T1GD-FLAIR":
        num_modalities = 2
        
        T1GD_file = dir_files[dir_files.index(patient_code + "_T1GD.nii")]
        FLAIR_file = dir_files[dir_files.index(patient_code + "_FLAIR.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T1GD_data = nib.load(os.path.join(path,T1GD_file)).get_fdata().astype('float32')
        FLAIR_data = nib.load(os.path.join(path,FLAIR_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T1GD_data, FLAIR_data], FA_data, num_modalities

    elif code == "T1-T2-T1GD":
        num_modalities = 3
        
        T1_file = dir_files[dir_files.index(patient_code +  "_T1.nii")]
        T2_file = dir_files[dir_files.index(patient_code + "_T2.nii")]
        T1GD_file = dir_files[dir_files.index(patient_code + "_T1GD.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T1_data = nib.load(os.path.join(path,T1_file)).get_fdata().astype('float32')
        T2_data = nib.load(os.path.join(path,T2_file)).get_fdata().astype('float32')
        T1GD_data = nib.load(os.path.join(path,T1GD_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T1_data, T2_data, T1GD_data], FA_data, num_modalities
    
    elif code == "T1-T2-FLAIR":
        num_modalities = 3
        
        T1_file = dir_files[dir_files.index(patient_code +  "_T1.nii")]
        T2_file = dir_files[dir_files.index(patient_code + "_T2.nii")]
        FLAIR_file = dir_files[dir_files.index(patient_code + "_FLAIR.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T1_data = nib.load(os.path.join(path,T1_file)).get_fdata().astype('float32')
        T2_data = nib.load(os.path.join(path,T2_file)).get_fdata().astype('float32')
        FLAIR_data = nib.load(os.path.join(path,FLAIR_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T1_data, T2_data, FLAIR_data], FA_data, num_modalities

    elif code == "T1-T1GD-FLAIR":
        num_modalities = 3
        
        T1_file = dir_files[dir_files.index(patient_code +  "_T1.nii")]
        T1GD_file = dir_files[dir_files.index(patient_code + "_T1GD.nii")]
        FLAIR_file = dir_files[dir_files.index(patient_code + "_FLAIR.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T1_data = nib.load(os.path.join(path,T1_file)).get_fdata().astype('float32')
        T1GD_data = nib.load(os.path.join(path,T1GD_file)).get_fdata().astype('float32')
        FLAIR_data = nib.load(os.path.join(path,FLAIR_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T1_data, T1GD_data, FLAIR_data], FA_data, num_modalities

    elif code == "T2-T1GD-FLAIR":
        num_modalities = 3
        
        T2_file = dir_files[dir_files.index(patient_code + "_T2.nii")]
        T1GD_file = dir_files[dir_files.index(patient_code + "_T1GD.nii")]
        FLAIR_file = dir_files[dir_files.index(patient_code + "_FLAIR.nii")]
        FA_file = dir_files[dir_files.index(patient_code + "_DTI_FA.nii")]
        
        # Load and retrieve fdata
        T2_data = nib.load(os.path.join(path,T2_file)).get_fdata().astype('float32')
        T1GD_data = nib.load(os.path.join(path,T1GD_file)).get_fdata().astype('float32')
        FLAIR_data = nib.load(os.path.join(path,FLAIR_file)).get_fdata().astype('float32')
        FA_data = nib.load(os.path.join(path,FA_file)).get_fdata().astype('float32')

        return [T2_data, T1GD_data, FLAIR_data], FA_data, num_modalities

    elif code == "all":
        num_modalities = 4
        
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

        return [T1_data, T2_data, T1GD_data, FLAIR_data], FA_data, num_modalities

def entropy_sort(num_modalities, MRI, FA, shann_threshold):
    
    # Set placeholder with padding
    strucMRI = np.zeros((256, 256, num_modalities)).astype('float32')
    FA_map = np.zeros((256, 256, num_modalities)).astype('float32')

    # Set placeholder for storing strucMRI images and FA images
    strucMRI_set = []    
    FA_set = []
    
    if num_modalities == 1:
        img = MRI[0]
                        
        for slice in range(0, 155):
            strucMRI_cpy = np.copy(strucMRI)
            FA_map_cpy = np.copy(FA_map)
            
            strucMRI_cpy[8:248, 8:248, 0] = img[:,:,slice] 
            FA_map_cpy[8:248, 8:248, 0] = FA[:,:,slice]
            
            # Normalise images between -1 and 1
            strucMRI_cpy = cv2.normalize(strucMRI_cpy, None, -1, 1, cv2.NORM_MINMAX)
            FA_map_cpy = cv2.normalize(FA_map_cpy, None, -1, 1, cv2.NORM_MINMAX)
            strucMRI_entropy = shannon_entropy(strucMRI_cpy)
            
            if strucMRI_entropy > shann_threshold:
                strucMRI_set.append([strucMRI_cpy[:,:,np.newaxis], strucMRI_entropy])
                FA_set.append([FA_map_cpy[:,:,np.newaxis], strucMRI_entropy])
            
            
    elif num_modalities > 1:        
        for slice in range(0, 155):
            strucMRI_cpy = np.copy(strucMRI)
            FA_map_cpy = np.copy(FA_map)
            
            for idx in range(0,len(MRI)):
                img = MRI[idx]
                strucMRI[8:248, 8:248, idx] = img[:,:,slice]
            FA_map[8:248, 8:248, 0] = FA[:,:,slice]
            
            # Normalise images between -1 and 1
            strucMRI_cpy = cv2.normalize(strucMRI_cpy, None, -1.0, 1.0, cv2.NORM_MINMAX)
            FA_map_cpy = cv2.normalize(FA_map_cpy, None, -1.0, 1.0, cv2.NORM_MINMAX)
            strucMRI_entropy = shannon_entropy(strucMRI_cpy)
            
            if strucMRI_entropy > shann_threshold:
                strucMRI_set.append([strucMRI_cpy, strucMRI_entropy])
                FA_set.append([FA_map_cpy, strucMRI_entropy])       
    
    return strucMRI_set, FA_set


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image, num_modalities):
    stacked_img = tf.concat([input_image, real_image], axis=2)
        
    cropped_image = tf.image.random_crop(
        stacked_img, size=[256, 256, num_modalities*2])
    
    return cropped_image[:,:,0:num_modalities], cropped_image[:,:,num_modalities:]


@tf.function()
def random_jitter(input_image, real_image, num_modalities):
    
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image, num_modalities)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path, code, shann_threshold):
    strucMRI, FA, num_modalities = load(idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path, code)
    input_imgs, real_imgs = entropy_sort(num_modalities, strucMRI, FA, shann_threshold)
    for img in range(0,len(input_imgs)):
        input_imgs[img][0], real_imgs[img][0] = random_jitter(input_imgs[img][0], real_imgs[img][0], num_modalities)
    return input_imgs, real_imgs


def load_image_test(idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path, code, shann_threshold):
    strucMRI, FA, num_modalities = load(idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path, code)
    input_imgs, real_imgs = entropy_sort(num_modalities, strucMRI, FA, shann_threshold)
    for img in range(0,len(input_imgs)):
        input_imgs[img][0], real_imgs[img][0] = resize(input_imgs[img][0], real_imgs[img][0],
                                   IMG_WIDTH, IMG_HEIGHT)
    return input_imgs, real_imgs


def train_dataset_init(train_dataset_lookup, test_dataset_lookup, train_path=TRAIN_PATH, test_path=TEST_PATH, code="all", shann_threshold=0.2):
    count = 1
    start = time.time()

    print(f"Generating training dataset with code {code}.\n")
    
    for idx, _ in train_dataset_lookup.items(): 
        if count == 1:
            input_set, target_set = load_image_train(idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path, code, shann_threshold)
            print(f"{count}/359.\n")
            count += 1
        else:
            input_imgs, target_imgs = load_image_train(idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path, code, shann_threshold)
            input_set = input_set + input_imgs
            target_set = target_set + target_imgs
            print(f"{count}/359.\n")
            count += 1
 
    input_set = [input_img[0] for input_img in sorted(input_set, key=lambda a: a[1])]
    target_set = [target_img[0] for target_img in sorted(target_set, key=lambda a: a[1])]

    train_input_dataset = tf.data.Dataset.from_tensor_slices(input_set)
    train_target_dataset = tf.data.Dataset.from_tensor_slices(target_set)
        
    # Create train dataset of source images (structural MRI)
    # train_input_dataset = train_input_dataset.shuffle(BUFFER_SIZE_TRAIN)
    train_input_dataset = train_input_dataset.batch(BATCH_SIZE)

    # Create train dataset of target images (FA)
    # train_target_dataset = train_target_dataset.shuffle(BUFFER_SIZE_TRAIN)
    train_target_dataset = train_target_dataset.batch(BATCH_SIZE)
    
    print(f"Training dataset generated after {time.time() - start} seconds.\n")
    print(f"Saving training data...")

    # Save training data
    path_train_input = os.path.join(os.getcwd(), os.path.join(code, "train_dataset_input"))
    path_train_target = os.path.join(os.getcwd(), os.path.join(code, "train_dataset_target"))
    train_input_dataset.save(path_train_input)
    train_target_dataset.save(path_train_target)

    print(f"Training data saved!\n")

    return (train_input_dataset, train_target_dataset)


def test_dataset_init(train_dataset_lookup, test_dataset_lookup, train_path=TRAIN_PATH, test_path=TEST_PATH, code="all", shann_threshold=0.2):  
    count = 1
    start = time.time()

    print(f"Generating test dataset with code {code}.\n")

    for idx, patient in test_dataset_lookup.items(): 
        if count == 1:
            input_set, target_set = load_image_test(idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path, code, shann_threshold)
            print(f"{count}/153.\n")
            count += 1
        else:
            input_imgs, target_imgs = load_image_test(idx, train_dataset_lookup, test_dataset_lookup, train_path, test_path, code, shann_threshold)
            input_set = input_set + input_imgs
            target_set = target_set + target_imgs
            print(f"{count}/153.\n")
            count += 1

    input_set = [input_img[0] for input_img in sorted(input_set, key=lambda a: a[1])]
    target_set = [target_img[0] for target_img in sorted(target_set, key=lambda a: a[1])]

    test_input_dataset = tf.data.Dataset.from_tensor_slices(input_set)
    test_target_dataset = tf.data.Dataset.from_tensor_slices(target_set)

    # Create train dataset of source images (structural MRI)
    # test_input_dataset = test_input_dataset.shuffle(BUFFER_SIZE_TEST)
    test_input_dataset = test_input_dataset.batch(BATCH_SIZE)

    # Create train dataset of target images (FA)
    # test_target_dataset = test_target_dataset.shuffle(BUFFER_SIZE_TEST)
    test_target_dataset = test_target_dataset.batch(BATCH_SIZE)
          
    print(f"Test dataset generated after {time.time() - start} seconds.\n")
    print(f"Saving test data...\n")

    # Save test data
    path_test_input = os.path.join(os.getcwd(), os.path.join(code, "test_dataset_input"))
    path_test_target = os.path.join(os.getcwd(), os.path.join(code, "test_dataset_target"))
    test_input_dataset.save(path_test_input)
    test_target_dataset.save(path_test_target)
    
    print(f"Test data saved!\n")
    
    return (test_input_dataset, test_target_dataset)

if __name__ == "__main__":
    
    # Set code
    code = "all"
    
    # Generate lookup table
    train_lookup, test_lookup = dataset_lookup_table_init(TRAIN_PATH, TEST_PATH)

    # Generate training dataset
    train_dataset = train_dataset_init(train_lookup, test_lookup, train_path=TRAIN_PATH, test_path=TEST_PATH, code=code, shann_threshold=0.4)
        
    # Generate test dataset
    test_dataset = test_dataset_init(train_lookup, test_lookup, train_path=TRAIN_PATH, test_path=TEST_PATH, code=code, shann_threshold=0.4)