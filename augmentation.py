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

def augment_data(arr: np.ndarray, gaussian):
    """Augment a 3D array by mirroring and rotating it and adding Gaussian noise."""
    l = data_transformation(arr)
    if gaussian:
        l = l + [add_gaussian(i) for i in l]
    else:
        l = l + [l[0]]
    return l

if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    train_dataset = torch.load('../train_dataset.pt')
    x_aug = []
    y_aug = []
    for x, y in tqdm(train_dataset):
        x = augment_data(x.unsqueeze(0).numpy(), gaussian=True)
        y = augment_data(y.unsqueeze(0).numpy(), gaussian=False)
        x_y = list(zip(x, y))
        x_y = random.sample(x_y, 4)
        x_aug += [i[0] for i in x_y]
        y_aug += [i[1] for i in x_y]
    x_aug = torch.tensor(np.concatenate(x_aug), dtype=float)
    y_aug = torch.tensor(np.concatenate(y_aug), dtype=float)
    train_augmented_dataset = torch.utils.data.TensorDataset(x_aug, y_aug)
    torch.save(train_augmented_dataset, '../train_dataset_augmented.pt')
