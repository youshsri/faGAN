import cv2
import numpy as np
from scipy import linalg
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input


def calculate_mae(img1: tf.Tensor, img2: tf.Tensor) -> float:
    """
    Computes the Mean Absolute Error (MAE) between two images.
    
    Args:
        img1: tf.Tensor, first grayscale image of dimensions 256 x 256 x 1.
        img2: tf.Tensor, second grayscale image of dimensions 256 x 256 x 1.
        
    Returns:
        The MAE value as a float between the two images.

    """
    # Ensure that the two images have the same shape
    assert img1.shape == img2.shape, "The two images must have the same shape."
    
    # Compute the MAE between the two images
    mae = np.mean(np.abs(img1.numpy() - img2.numpy()))
    
    return mae


def calculate_rmse(img1: tf.Tensor, img2: tf.Tensor) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between two images.

    Args:
        img1: tf.Tensor, first grayscale image of dimensions 256 x 256 x 1.
        img2: tf.Tensor, second grayscale image of dimensions 256 x 256 x 1.
        
    Returns:
        The RSME value as a float between the two images.
    
    """
    # Ensure that the two images have the same shape
    assert img1.shape == img2.shape, "The two images must have the same shape."
    
    # Compute the RMSE between the two images
    mse = np.mean((img1.numpy() - img2.numpy()) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def calculate_psnr(img1: tf.Tensor, img2: tf.Tensor) -> float:
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two grayscale images.
    
    Args:
        img1: tf.Tensor, first grayscale image of dimensions 256 x 256 x 1.
        img2: tf.Tensor, second grayscale image of dimensions 256 x 256 x 1.
        
    Returns:
        The PSNR value as a float.
    """

    # Convert intensities to range of 0, 255 
    img1 = (img1.numpy()*0.5 + 0.5) * 255
    img2 = (img2.numpy()*0.5 + 0.5) * 255
    
    # Compute MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # Compute PSNR
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    
    return psnr


def calculate_ssim(img1: tf.Tensor, img2: tf.Tensor) -> float:
    """Calculate the Structural Similarity Index (SSIM) between two grayscale images.
    
    Args:
        img1: numpy.ndarray, first grayscale image of dimensions 256 x 256 x 1.
        img2: numpy.ndarray, second grayscale image of dimensions 256 x 256 x 1.
        
    Returns:
        The SSIM value as a float between -1 and 1, where 1 indicates a perfect match.
    """

    # Convert intensities to range of 0, 255 
    img1 = (img1.numpy()*0.5 + 0.5) * 255
    img2 = (img2.numpy()*0.5 + 0.5) * 255
    
    # Compute means, variances, and covariance
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5, borderType=cv2.BORDER_REPLICATE)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5, borderType=cv2.BORDER_REPLICATE) - mu1_mu2
    
    # Compute SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Compute SSIM numerator and denominator
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    # Compute SSIM
    ssim_map = numerator / denominator
    ssim = np.mean(ssim_map)
    
    return ssim


def calculate_metrics(img1: tf.Tensor, img2: tf.Tensor):
    """Calculate metrics between two grayscale images.
    
    Args:
        img1: tf.Tensor, first grayscale image of dimensions 256 x 256 x 1.
        img2: tf.Tensor, second grayscale image of dimensions 256 x 256 x 1.
        
    Returns:
        The calculated metrics between the two images, including:
            - SSIM
            - PSNR
            - RSME
            - MAE
    """
    SSIM = calculate_ssim(img1, img2)
    PSNR = calculate_psnr(img1, img2)
    RSME = calculate_rmse(img1, img2)
    MAE = calculate_mae(img1, img2)

    return [SSIM, PSNR, RSME, MAE]

def shannon_entropy(img):
    """
    Calculates the Shannon-entropy of an image.

    Parameters:
        img: numpy array of image

    Returns:
        entropy: float value representing the Shannon-entropy of the image
    """
    img = np.asarray(img, dtype=np.uint32).ravel()
    
    # Count the frequency of each pixel value
    counts = Counter(img)
   
    # Calculate the probability of each pixel value
    probabilities = [float(c) / len(img) for c in counts.values()]
    
    # Calculate the entropy
    entropy = - sum([p * np.log2(p) for p in probabilities if p > 0])
    
    return entropy

if __name__ == "__main__":
    pass