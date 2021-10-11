import os
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold, StratifiedKFold
from pytorch_dataset import *

# Configuration options
k_folds = 5

# Set fixed random number seed
torch.manual_seed(42)

# Prepare MNIST dataset by concatenating Train/Test part; we split later.
test_transform = test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=5,scale=(0.45,1.25),translate=(0.3,0.3))
        ])
dataset = ObjectClassifierDataset("/home/geffen/Desktop/Face_Detector/assemble_face_dataset_utils/face_classifier_dataset/train",test_transform)

# Define the K-fold Cross Validator
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

# Start print
print('--------------------------------')

# K-fold Cross Validation model evaluation
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset,dataset.labels)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    
    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=128, sampler=train_subsampler)
    val_loader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=128, sampler=val_subsampler)
    
    
    print(f'\nFOLD {fold}')
    print('--------------------------------')
    print(len(val_loader))
    # print(len(train_ids))
    # print(len(val_ids))
    # for i,batch in enumerate(val_loader):
    #     print(i)
