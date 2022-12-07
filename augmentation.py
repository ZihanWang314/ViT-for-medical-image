import numpy as np
import random
# import matplotlib.pyplot as plt
# import nibabel as nib

# Augmentation functions
def data_transformation(arr: np.ndarray):
    """Augment 3D arrays by mirroring and rotating them."""
    # declare result: list of nd.array
    assert len(arr.shape) == 4
    result = [arr]

    # mirror along x axis
    result.append(np.flip(arr, axis=1))

    # mirror along y axis
    result.append(np.flip(arr, axis=2))

    # mirror along z axis
    result.append(np.flip(arr, axis=3))

    # rotate along x axis
    result.append(np.rot90(arr, k=1, axes=(2, 3)))
    result.append(np.rot90(arr, k=2, axes=(2, 3)))
    result.append(np.rot90(arr, k=3, axes=(2, 3)))

    # rotate along y axis
    result.append(np.rot90(arr, k=1, axes=(1, 3)))
    result.append(np.rot90(arr, k=2, axes=(1, 3)))
    result.append(np.rot90(arr, k=3, axes=(1, 3)))

    # rotate along z axis
    result.append(np.rot90(arr, k=1, axes=(1, 2)))
    result.append(np.rot90(arr, k=2, axes=(1, 2)))
    result.append(np.rot90(arr, k=3, axes=(1, 2)))
    
    return result

def add_gaussian(arr: np.ndarray):
    """Add Gaussian noise to a 3D array."""
    return arr + np.random.normal(0, 0.01, arr.shape)

def augment_data(arr: np.ndarray):
    """Augment a 3D array by mirroring and rotating it and adding Gaussian noise."""
    l = data_transformation(arr)
    l = l + [add_gaussian(i) for i in l]
    return random.sample(l, 4)