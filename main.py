import numpy as np
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
from random import shuffle

# from tqdm.notebook import trange, tqdm
from tqdm import tqdm, trange
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
import argparse

from utils import patchify, unpatchify, random_masking, restore_masked
from models import Transformer, MaskedAutoEncoder, SegmentationMAE

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
args = parser.parse_args()



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_folder = r".."

batch_size = 32

trainset = torch.load(os.path.join(root_folder,'train_dataset.pt'))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torch.load(os.path.join(root_folder, 'dev_dataset.pt'))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
                                        

if args.task == 'mae':
    # Initilize MAE model
    model = MaskedAutoEncoder(
        Transformer(embedding_dim=256, n_layers=6),
        Transformer(embedding_dim=128, n_layers=6),
        patch_size=8,
        num_patches=64
    )
    # Move the model to GPU
    model.to(torch_device)
    # Create optimizer

    # You may want to tune these hyperparameters to get better performance
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)

    total_steps = 0
    num_epochs = 5
    train_logfreq = 100

    losses = []

    epoch_iterator = trange(num_epochs)
    for epoch in epoch_iterator:
        # Train
        data_iterator = tqdm(trainloader)
        for x, y in data_iterator:
            total_steps += 1
            x = x.to(torch_device)
            image_patches = patchify(x, patch_size=model.patch_size)
            predicted_patches, mask = model(x)
            loss = torch.sum(torch.mean(torch.square(image_patches - predicted_patches), dim=-1) * mask) / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.set_postfix(loss=loss.item())
            if total_steps % train_logfreq == 0:
                losses.append(loss.item())

        # Periodically save model
        torch.save(model.state_dict(), os.path.join(root_folder, "mae_pretrained.pt"))

    plt.plot(losses)
    plt.title('MAE Train Loss')
    plt.savefig('mae_losses')

elif args.task == 'seg':
    mae = MaskedAutoEncoder(
        Transformer(embedding_dim=256, n_layers=6),
        Transformer(embedding_dim=128, n_layers=6),
        patch_size=8,
        num_patches=64
    )
    mae.load_state_dict(torch.load(os.path.join(root_folder, "mae_pretrained.pt")))

    # Initilize classification model; set detach=True to only update the linear classifier. 
    model = SegmentationMAE(mae, detach=True)
    model.to(torch_device)

    # You may want to tune these hyperparameters to get better performance
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-9)

    total_steps = 0
    num_epochs = 5
    train_logfreq = 100
    losses = []
    train_acc = []
    all_val_acc = []
    best_val_acc = 0

    epoch_iterator = trange(num_epochs)
    for epoch in epoch_iterator:
        # Train
        data_iterator = tqdm(trainloader)
        for x, y in data_iterator:
            total_steps += 1
            x, y = x.to(torch_device), y.to(torch_device).float()
            logits = model(x)
            loss = torch.mean(F.binary_cross_entropy_with_logits(logits, y))
            accuracy = ((logits > 0) & (y == 1)).count_nonzero() / ((logits > 0) | (y == 1)).count_nonzero()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.set_postfix(loss=loss.item(), train_acc=accuracy.item())

            if total_steps % train_logfreq == 0:
                losses.append(loss.item())
                train_acc.append(accuracy.item())

        # Validation
        val_acc = []
        model.eval()
        for x, y in testloader:
            x, y = x.to(torch_device), y.to(torch_device).float()
            with torch.no_grad():
                logits = model(x)
            accuracy = ((logits > 0) & (y == 1)).count_nonzero() / ((logits > 0) | (y == 1)).count_nonzero()
            val_acc.append(accuracy.item())

        model.train()

        all_val_acc.append(np.mean(val_acc))

        # Save best model
        if np.mean(val_acc) > best_val_acc:
            best_val_acc = np.mean(val_acc)

        epoch_iterator.set_postfix(val_acc=np.mean(val_acc), best_val_acc=best_val_acc)

    plt.plot(losses)
    plt.title('Linear Classification Train Loss')
    plt.savefig('seg_losses')
    plt.clf()
    plt.plot(train_acc)
    plt.title('Linear Classification Train Accuracy')
    plt.savefig('seg_train_acc')
    plt.clf()
    plt.plot(all_val_acc)
    plt.title('Linear Classification Val Accuracy')
    plt.savefig('seg_val_acc')
    plt.clf()
