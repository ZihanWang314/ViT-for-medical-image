import numpy as np
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
from random import shuffle

from tqdm.notebook import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import einops
# import pickle
import os
# import nibabel as nib
# import skimage.transform as skTrans
import numpy as np


def patchify(images, patch_size=4):
    """Splitting images into patches.
    Args:
        images: Input tensor with size (batch, channels, height, width)
            We can assume that image is square where height == width.
    Returns:
        A batch of image patches with size (
          batch, (height / patch_size) * (width / patch_size), 
        channels * patch_size * patch_size)
    Hint: use einops.rearrange. The "space-to-depth operation" example at https://einops.rocks/api/rearrange/ 
    is not exactly what you need, but it gives a good idea of how to use rearrange.
    """
    return einops.rearrange(
        images,
        'b (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3)',
        p1=patch_size,
        p2=patch_size,
        p3=patch_size
    )

def unpatchify(patches, patch_size=4):
    """Combining patches into images.
    Args:
        patches: Input tensor with size (
        batch, (height / patch_size) * (width / patch_size), 
        channels * patch_size * patch_size)
    Returns:
        A batch of images with size (batch, channels, height, width)
        
    Hint: einops.rearrange can be used here as well.
    """
    return einops.rearrange(
        patches,
        'b (h w d) (p1 p2 p3) -> b (h p1) (w p2) (d p3)',
        p1=patch_size,
        p2=patch_size,
        p3=patch_size,
        h=32 // patch_size,
        w=32 // patch_size,
        d=32 // patch_size
    )


def index_sequence(x, ids):
    """Index tensor (x) with indices given by ids
    Args:
        x: input sequence tensor, can be 2D (batch x length) or 3D (batch x length x feature)
        ids: 2D indices (batch x length) for re-indexing the sequence tensor
    """
    if len(x.shape) == 3:
        ids = ids.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.take_along_dim(x, ids, dim=1)

def random_masking(x, keep_length, ids_shuffle):
    """Apply random masking on input tensor
    Args:
        x: input patches (batch x length x feature)
        keep_length: length of unmasked patches
        ids_shuffle: random indices for shuffling the input sequence. This is an
            array of size (batch x length) where each row is a permutation of 
            [0, 1, ..., length-1]. We will pass this array to index_sequence function
            to chooose the unmasked patches.
    Returns:
        kept: unmasked part of x
        mask: a 2D (batch x length) mask tensor of 0s and 1s indicated which
            part of x is masked out. The value 0 indicates not masked and 1
            indicates masked.
        ids_restore: indices to restore x. This is an array of size (batch x length).
            If we take the kept part and masked
            part of x, concatentate them together and index it with ids_restore,
            we should get x back. (Hint: try using torch.argsort on the shuffle indices)

    Hint:
        ids_shuffle contains the indices used to shuffle the sequence (patches).
        You should use the provided index_sequence function to re-index the
        sequence, and keep the first keep_length number of patches.
    """
    # BEGIN YOUR CODE
    ids_broadcast = ids_shuffle[:, :, None].repeat(1, 1, x.shape[2])
    x_shuffled = torch.gather(x, 1, ids_broadcast)
    kept = x_shuffled[:, :keep_length]
    ids_restore = torch.argsort(ids_shuffle)
    mask = torch.zeros_like(ids_shuffle)
    mask[:, :keep_length] = 1
    mask = 1 - torch.gather(mask, 1, ids_restore)

    return kept, mask, ids_restore


    # END YOUR CODE

def restore_masked(kept_x, masked_x, ids_restore):
    """Restore masked patches
    Args:
        kept_x: unmasked patches: (batch x keep_length x feature)
        masked_x: masked patches: (batch x (length - keep_length) x feature)
        ids_restore: indices to restore x: (batch x length)
    Returns:
        restored patches
    Hint: call the index_sequence function on an array with the kept and masked tokens concatenated
    """
    # BEGIN YOUR CODE
    all_x = torch.concat([kept_x, masked_x], dim=1)
    restored_x = torch.gather(all_x, 1, ids_restore[:, :, None].repeat(1, 1, all_x.shape[2]))
    return restored_x

    # END YOUR CODE