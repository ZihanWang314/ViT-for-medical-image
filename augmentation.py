import numpy as np
# import matplotlib.pyplot as plt
# import nibabel as nib

# Augmentation functions
def data_transformation(arr: np.ndarray):
    """Augment a 3D array by mirroring and rotating it."""
    # declare result: list of nd.array
    result = [arr]

    # mirror along x axis
    result.append(np.flip(arr, axis=0))

    # mirror along y axis
    result.append(np.flip(arr, axis=1))

    # mirror along z axis
    result.append(np.flip(arr, axis=2))

    # rotate along x axis
    result.append(np.rot90(arr, k=1, axes=(1, 2)))
    result.append(np.rot90(arr, k=2, axes=(1, 2)))
    result.append(np.rot90(arr, k=3, axes=(1, 2)))

    # rotate along y axis
    result.append(np.rot90(arr, k=1, axes=(0, 2)))
    result.append(np.rot90(arr, k=2, axes=(0, 2)))
    result.append(np.rot90(arr, k=3, axes=(0, 2)))

    # rotate along z axis
    result.append(np.rot90(arr, k=1, axes=(0, 1)))
    result.append(np.rot90(arr, k=2, axes=(0, 1)))
    result.append(np.rot90(arr, k=3, axes=(0, 1)))
    
    return result

def add_gaussian(arr: np.ndarray):
    """Add Gaussian noise to a 3D array."""
    return arr + np.random.normal(0, 0.01, arr.shape)

def augment_data(arr: np.ndarray):
    """Augment a 3D array by mirroring and rotating it and adding Gaussian noise."""
    l = data_transformation(arr)
    return l + [add_gaussian(i) for i in l]