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

from utils import patchify, unpatchify, random_masking, restore_masked

class Transformer(nn.Module):
    """Transformer Encoder 
    Args:
        embedding_dim: dimension of embedding
        n_heads: number of attention heads
        n_layers: number of attention layers
        feedforward_dim: hidden dimension of MLP layer
    Returns:
        Transformer embedding of input
    """
    def __init__(self, embedding_dim=256, n_heads=4, n_layers=4, feedforward_dim=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.feedforward_dim = feedforward_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=self.n_heads,
                dim_feedforward=self.feedforward_dim,
                activation=F.gelu,
                batch_first=True,
                dropout=0.0,
            ),
            num_layers=n_layers,
        )

    def forward(self, x):
        return self.transformer(x)



class MaskedAutoEncoder(nn.Module):
    """MAE Encoder
    Args:
        encoder: vit encoder
        decoder: vit decoder
        encoder_embedding_dim: embedding size of encoder
        decoder_embedding_dim: embedding size of decoder
        patch_size: image patch size
        num_patches: number of patches
        mask_ratio: percentage of masked patches
    """
    def __init__(self, encoder, decoder, encoder_embedding_dim=256, 
                 decoder_embedding_dim=128, patch_size=4, num_patches=8,
                 mask_ratio=0.75):
        super().__init__()
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio

        self.masked_length = int(num_patches * mask_ratio)
        self.keep_length = num_patches - self.masked_length

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_input_projection = nn.Linear(patch_size * patch_size * patch_size, encoder_embedding_dim)
        self.decoder_input_projection = nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        self.decoder_output_projection = nn.Linear(decoder_embedding_dim, patch_size * patch_size * patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embedding_dim) * 0.02)
        self.encoder_position_encoding = nn.Parameter(torch.randn(1, num_patches, encoder_embedding_dim) * 0.02)
        self.decoder_position_encoding = nn.Parameter(torch.randn(1, num_patches, decoder_embedding_dim) * 0.02)
        self.masked_tokens = nn.Parameter(torch.randn(1, 1, decoder_embedding_dim) * 0.02)

    def forward_encoder(self, images, ids_shuffle=None):
        """Encode input images
        You should implement the following steps
        (1) patchify images into patches
        (2) linear projection
        (3) add position encoding
        (4) mask out some of the patches
        (5) concatenate cls_token and patches embedding and pass it to vit encoder
        """
        batch_size = images.shape[0]
        # Generate random shuffling indices
        if ids_shuffle is None:
            ids_shuffle = torch.argsort(
                torch.rand(
                    (batch_size, self.num_patches),
                    device=images.device
                ),
                dim=1
            )
        # BEGIN YOUR CODE
        patched_images = patchify(images, patch_size=self.patch_size)
        embedded_images = self.encoder_input_projection(patched_images)
        embedded_images += self.encoder_position_encoding
        kept, mask, ids_restore = random_masking(embedded_images, self.keep_length, ids_shuffle)
        encoder_input = torch.concat([self.cls_token.repeat(batch_size, 1, 1), kept], dim=1)
        encoder_output = self.encoder(encoder_input)
        return encoder_output, mask, ids_restore

        # END YOUR CODE

    def forward_decoder(self, encoder_embeddings, ids_restore):
        """Decode encoder embeddings
        You should implement the following steps
        (1) linear projection of encoder embeddings
        (2) restore sequence from masked_patches and encoder predictions
            We need our decoder output to have the same number of tokens as the original patched input, 
            but our encoder currently is missing some tokens that we masked out. We need to add some dummy 
            tokens back to our embedded tokens so when our decoder outputs, it has the correct number of 
            tokens that can be interpreted as sequential patches of output image.
            During this step, you should remove the CLS token from the sequence.
        (3) add position encoding
        (3) re-concatenate/use CLS token and decode using ViT decoder 
        (4) projection to predict image patches
        """
        # BEGIN YOUR CODE
        batch_size, sequence_length = ids_restore.shape
        decoder_projected = self.decoder_input_projection(encoder_embeddings)
        cls_token = decoder_projected[:, 0:1]
        projected = decoder_projected[:, 1:]
        dummy_tokens = self.masked_tokens.repeat(batch_size, sequence_length - self.keep_length, 1)
        restored = restore_masked(projected, dummy_tokens, ids_restore)
        decoder_input = restored + self.decoder_position_encoding
        decoder_input = torch.concat([cls_token, decoder_input], dim=1)
        decoder_output = self.decoder(decoder_input)[:, 1:]
        decoder_output = self.decoder_output_projection(decoder_output)
        return decoder_output
        
        # END YOUR CODE

    def forward(self, images):
        encoder_output, mask, ids_restore = self.forward_encoder(images)
        decoder_output = self.forward_decoder(encoder_output, ids_restore)
        return decoder_output, mask

    def forward_encoder_representation(self, images):
        """Encode images without applying random masking to get representation
        of input images. 

        You should implement splitting images into patches, re-concatenate/use CLS token,
        and encoding with ViT encoder.
        """
        # BEGIN YOUR CODE
        batch_size = images.shape[0]
        patched_images = patchify(images, patch_size=self.patch_size)
        embedded_images = self.encoder_input_projection(patched_images)
        embedded_images += self.encoder_position_encoding
        encoder_input = torch.concat([self.cls_token.repeat(batch_size, 1, 1), embedded_images], dim=1)
        encoder_output = self.encoder(encoder_input)
        return encoder_output
        # END YOUR CODE


class SegmentationMAE(nn.Module):
    """A linear classifier is trained on self-supervised representations learned by MAE. 
    Args:
        n_classes: number of classes
        mae: mae model
        embedding_dim: embedding dimension of mae output
        detach: if True, only the classification head is updated.
    """
    def __init__(self, mae, embedding_dim=256, detach=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mae = mae
        self.mae.mask_ratio = 0
        """
        When self.detach=True, use linear classification, when self.detach=False,
        use full finetuning.
        """
        self.detach = detach

    def forward(self, images):
        """
        Args:
            Images: batch of images
        Returns:
            logits: batch of logits from the ouput_head
        Remember to detach the representations if self.detach=True, and 
        Remember that we do not use masking here.
        """
        # BEGIN YOUR CODE
        mae_output = self.mae(images)[0]
        output = unpatchify(mae_output, 8)
        return output
        # END YOUR CODE


