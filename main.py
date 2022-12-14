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
from augmentation import augment_data

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
from models import Transformer, MaskedAutoEncoder, SegmentationMAE, RegressionMAE

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--encoder_dim', type=int, default=512)
parser.add_argument('--decoder_dim', type=int, default=512)
parser.add_argument('--ff_dim', type=int, default=2048)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoches', type=int, default=100)
parser.add_argument('--exp_name', type=str, default='exp_log')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--load_pretraining', action='store_true')
parser.add_argument('--data_augmentation', action='store_true')
args = parser.parse_args()


if not os.path.exists(args.exp_name):
    os.mkdir(args.exp_name)


torch_device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
print(f'using device: cuda:{args.gpu_id}')
root_folder = r".."

batch_size = args.batch_size

                                        

if args.task == 'mae':
    if args.data_augmentation:
        trainset = torch.load(os.path.join(root_folder,'train_dataset_augmented.pt'))
    else:
        trainset = torch.load(os.path.join(root_folder,'train_dataset.pt'))
    testset = torch.load(os.path.join(root_folder, 'dev_dataset.pt'))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    # Initilize MAE model
    model = MaskedAutoEncoder(
        Transformer(embedding_dim=args.encoder_dim, n_heads=args.n_heads, n_layers=args.n_layers, feedforward_dim=args.ff_dim),
        Transformer(embedding_dim=args.decoder_dim, n_heads=args.n_heads, n_layers=args.n_layers, feedforward_dim=args.ff_dim),
        encoder_embedding_dim=args.encoder_dim, 
        decoder_embedding_dim=args.decoder_dim,
        patch_size=8,
        num_patches=64
    )
    # Move the model to GPU
    model.to(torch_device)
    model.train()

    # Create optimizer

    # You may want to tune these hyperparameters to get better performance
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    total_steps = 0
    num_epochs = args.num_epoches
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
        torch.save(model.state_dict(), os.path.join(args.exp_name, "mae_pretrained.pt"))

    plt.plot(losses)
    plt.title('MAE Train Loss')
    plt.savefig(os.path.join(args.exp_name, 'mae_losses'))
    print('plot has been saved to mae_losses.png')

elif args.task == 'seg':
    if args.data_augmentation:
        trainset = torch.load(os.path.join(root_folder,'train_dataset_augmented.pt'))
    else:
        trainset = torch.load(os.path.join(root_folder,'train_dataset.pt'))
    testset = torch.load(os.path.join(root_folder, 'dev_dataset.pt'))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    mae = MaskedAutoEncoder(
        Transformer(embedding_dim=args.encoder_dim, n_heads=args.n_heads, n_layers=args.n_layers, feedforward_dim=args.ff_dim),
        Transformer(embedding_dim=args.decoder_dim, n_heads=args.n_heads, n_layers=args.n_layers, feedforward_dim=args.ff_dim),
        encoder_embedding_dim=args.encoder_dim, 
        decoder_embedding_dim=args.decoder_dim,
        patch_size=8,
        num_patches=64
    )
    if args.load_pretraining:
        pass
        # mae.load_state_dict(torch.load(os.path.join(args.exp_name, "mae_pretrained.pt")))

    # Initilize classification model; set detach=True to only update the linear classifier. 
    model = SegmentationMAE(mae, detach=True)
    # if args.load_pretraining:
    #     model.load_state_dict(torch.load(os.path.join(root_folder, "seg_best.pt")))
    model.to(torch_device)
    model.train()

    # You may want to tune these hyperparameters to get better performance
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    total_steps = 0
    num_epochs = args.num_epoches
    train_logfreq = 100
    losses = []
    train_dice = []
    all_val_dice = []
    best_val_dice = 0

    epoch_iterator = trange(num_epochs)
    for epoch in epoch_iterator:
        # Train
        model.train()
        if epoch == num_epochs // 2:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-5)
            print('changing optimizer')
        data_iterator = tqdm(trainloader)
        for x, y in data_iterator:
            total_steps += 1
            x, y = x.to(torch_device).float(), y.to(torch_device).float()

            logits = model(x)
            loss = torch.mean(F.binary_cross_entropy_with_logits(logits, y))
            iou = ((logits > 0) & (y == 1)).count_nonzero() / ((logits > 0) | (y == 1)).count_nonzero()
            dice = 2 * iou / (iou + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.set_postfix(loss=loss.item(), train_dice=dice.item())

            if total_steps % train_logfreq == 0:
                losses.append(loss.item())
                train_dice.append(dice.item())

        # Validation
        val_dice = []
        model.eval()
        for x, y in testloader:
            x, y = x.to(torch_device).float(), y.to(torch_device).float()
            with torch.no_grad():
                logits = model(x)
            iou = ((logits > 0) & (y == 1)).count_nonzero() / ((logits > 0) | (y == 1)).count_nonzero()
            dice = 2 * iou / (iou + 1)
            val_dice.append(dice.item())

        all_val_dice.append(np.mean(val_dice))

        # Save best model
        if np.mean(val_dice) > best_val_dice:
            best_val_dice = np.mean(val_dice)
            torch.save(model.state_dict(), os.path.join(root_folder, "seg_best.pt"))

        epoch_iterator.set_postfix(val_dice=np.mean(val_dice), best_val_dice=best_val_dice)

    if args.load_pretraining:
        plt.plot(losses)
        plt.title('Linear Classification Train Loss')
        plt.savefig(os.path.join(args.exp_name, 'seg_losses'))
        plt.clf()
        plt.plot(train_dice)
        plt.title('Linear Classification Train dice')
        plt.savefig(os.path.join(args.exp_name, 'seg_train_dice'))
        plt.clf()
        plt.plot(all_val_dice)
        plt.title('Linear Classification Val dice')
        plt.savefig(os.path.join(args.exp_name, 'seg_val_dice'))
        plt.clf()
        print('plots have been saved to seg_losses.png, seg_train_dice.png and seg_val_dice.png')
    else:
        plt.plot(losses)
        plt.title('Linear Classification Train Loss')
        plt.savefig(os.path.join(args.exp_name, 'seg_losses_nopt'))
        plt.clf()
        plt.plot(train_dice)
        plt.title('Linear Classification Train dice')
        plt.savefig(os.path.join(args.exp_name, 'seg_train_dice_nopt'))
        plt.clf()
        plt.plot(all_val_dice)
        plt.title('Linear Classification Val dice')
        plt.savefig(os.path.join(args.exp_name, 'seg_val_dice_nopt'))
        plt.clf()
        print('plots have been saved to seg_losses_nopt.png, seg_train_dice_nopt.png and seg_val_dice_nopt.png')



elif args.task == 'reg':
    if args.data_augmentation:
        trainset = torch.load(os.path.join(root_folder,'train_dataset_regression.pt'))
    else:
        trainset = torch.load(os.path.join(root_folder,'train_dataset.pt'))
    testset = torch.load(os.path.join(root_folder, 'dev_dataset.pt'))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    mae = MaskedAutoEncoder(
        Transformer(embedding_dim=args.encoder_dim, n_heads=args.n_heads, n_layers=args.n_layers, feedforward_dim=args.ff_dim),
        Transformer(embedding_dim=args.decoder_dim, n_heads=args.n_heads, n_layers=args.n_layers, feedforward_dim=args.ff_dim),
        encoder_embedding_dim=args.encoder_dim, 
        decoder_embedding_dim=args.decoder_dim,
        patch_size=8,
        num_patches=64
    )
    if args.load_pretraining:
        mae.load_state_dict(torch.load(os.path.join(args.exp_name, "mae_pretrained.pt")))

    # Initilize classification model; set detach=True to only update the linear classifier. 
    model = RegressionMAE(mae, detach=True)
    model.to(torch_device)

    # You may want to tune these hyperparameters to get better performance
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    total_steps = 0
    num_epochs = args.num_epoches
    train_logfreq = 100
    losses_reg = []
    losses_class = []
    val_losses_reg = []
    val_accuracy = []

    epoch_iterator = trange(num_epochs)
    for epoch in epoch_iterator:
        # Train
        model.train()
        if epoch == num_epochs // 2:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-5)
            print('changing optimizer')
        data_iterator = tqdm(trainloader)
        for x, y in data_iterator:
            total_steps += 1
            y_reg, y_class = y
            x, y_reg, y_class = x.to(torch_device).float(), y_reg.to(torch_device).float(), y_class.to(torch_device).float()

            output_reg, output_class = model(x)
            loss_reg = torch.mean(F.mse_loss(output_reg, y_reg))
            loss_class = torch.mean(F.binary_cross_entropy_with_logits(output_class, y_class))
            loss = (loss_reg + loss_class) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.set_postfix(loss=loss.item())

            if total_steps % train_logfreq == 0:
                losses_reg.append(loss.item())
                losses_class.append(loss.item())

        # Validation
        losses_reg = []
        losses_class = []
        accuracies = []
        model.eval()
        for x, y in testloader:
            x, y = x.to(torch_device).float(), y.to(torch_device).float()
            with torch.no_grad():
                output_reg, output_class = model(x)
            loss_reg = torch.mean(F.mse_loss(output_reg, y_reg))
            accuracy = torch.mean(((output_class > 0).bool() == y_class.bool()).float())
            losses_reg.append(loss_reg)
            accuracies.append(accuracy)

        losses_reg = sum(losses_reg) / len(losses_reg)
        losses_class = sum(losses_class) / len(losses_class)
        accuracies = sum(accuracies) / len(accuracies)
        val_losses_reg.append(losses_reg)
        val_accuracy.append(accuracies.tolist())

        # Save best model

        epoch_iterator.set_postfix(loss_reg =losses_reg.tolist(), accuracy=accuracies.tolist())
    losses_class = []
    val_losses_reg = []
    val_accuracy = []

    plt.plot(losses_reg)
    plt.title('Regression losses for training')
    plt.savefig(os.path.join(args.exp_name, 'reg_loss_train'))
    plt.clf()
    plt.plot(losses_class)
    plt.title('Classification losses for training')
    plt.savefig(os.path.join(args.exp_name, 'class_loss_train'))
    plt.clf()
    plt.plot(val_losses_reg)
    plt.title('Validation MSE for regression')
    plt.savefig(os.path.join(args.exp_name, 'reg_mse_val'))
    plt.clf()
    plt.plot(val_accuracy)
    plt.title('Validation accuracy for bifurcation checking')
    plt.savefig(os.path.join(args.exp_name, 'reg_acc_val'))
    plt.clf()
    print('plots have been saved')
