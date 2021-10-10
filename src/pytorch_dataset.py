'''
pytorch_dataset.py
This file contains functions and classes for
creating a pytorch dataset and dataloader
'''

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from skimage import io, transform
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image,ImageDraw
import torch
import pandas as pd
import cv2
import random


# ===================================================================================================== #
# ============================================= CLASSES =============================================== #
# ===================================================================================================== #
'''
Generic Class for creating object classifier datasets
Parameters:
  img_dir_path - Full path to directory with the images for this dataset.
                 This assumes that the subdirectories contain each class, 
                 only images are in these subdirectories, and that the
                 subdirectory basenames are the desired name of the object class.
                 i.e. dog/dog1.png, cat/cat1.png, etc.

  transform -    Specifies the image format (size, RGB, etc.) and augmentations to use
'''
class ObjectClassifierDataset(Dataset):
    def __init__(self,img_dir_path,transform):
        self.img_dir_path = img_dir_path
        self.transform = transform
        
        # collect img classes from dir names
        img_classes = next(os.walk(img_dir_path))[1]
        
        # generate a dictionary to map class names to integers idxs
        self.labels = {img_classes[i] : i for i in range(0, len(img_classes))}
        
        # get all training samples by getting paths of the images in each subfolder folder
        self.imgs = []
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            if idx > 0: # we don't want the files in the top folder
                for file in path_obj[2]: # path_obj[2] is list of files in the subdirectory
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path 
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # load the image
        img = io.imread(self.imgs[idx])
        
        # apply any transformation
        if self.transform:
            img = self.transform(img)
        
        # return the sample (img (tensor)), object class (int), sample index (int))
        return img, self.labels[os.path.basename(os.path.dirname(self.imgs[idx]))], idx
    
    # Displays a random batch of 64 samples
    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size,shuffle=True)
        
        # get the first batch
        (imgs, labels, idxs) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        obj_classes = list(self.labels)
        
        fig,ax_array = plt.subplots(8,8,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                text = str(labels[idx].item()) + ":" + obj_classes[labels[idx]]  + ", i=" +str(idxs[idx].item())
                ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0), cmap="gray")
                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()



# ===================================================================================================== #
# ============================================= MAIN ================================================== #
# ===================================================================================================== #

if __name__  == "__main__":
    # define training data format
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1)
        ])
    img_data_dir1 = "/home/geffen/Desktop/Face_Detector/assemble_face_dataset_utils/face_classifier_dataset/test"
    faces_dataset = ObjectClassifierDataset(img_data_dir1,test_transform)
    faces_dataset.visualize_batch()