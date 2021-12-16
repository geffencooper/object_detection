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
import warnings
#warnings.filterwarnings("error")
torch.manual_seed(42)

# ===================================================================================================== #
# ========================================= DATASET CLASSES =========================================== #
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
    def __init__(self,img_dir_path,transform,normalize):
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.normalize = normalize
        
        # collect img classes from dir names
        img_classes = next(os.walk(img_dir_path))[1]
        
        # generate a dictionary to map class names to integers idxs
        self.classes = {img_classes[i] : i for i in range(0, len(img_classes))}
        
        # get all training samples/labels by getting paths of the images in each subfolder folder
        self.imgs = []
        self.labels = []
        i = 0
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            if idx > 0: # we don't want the files in the top folder
                for file in path_obj[2]: # path_obj[2] is list of files in the subdirectory
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path
                    self.labels.append(self.classes[os.path.basename(os.path.dirname(self.imgs[i]))]) # get the label from the directory name
                    i+=1
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # load the image
        try:
            img = io.imread(self.imgs[idx])
            
            # apply any transformation
            if self.transform:
                img = self.transform(img)
                if self.normalize:
                    norm = torchvision.transforms.Normalize((torch.mean(img)),(torch.std(img)))
                    img = norm(img)
                    
            
            # get the label
            label = self.labels[idx]
            
            # return the sample (img (tensor)), object class (int), sample index (int))
            return img, label, idx
        except (ValueError, RuntimeWarning,UserWarning) as e:
            print("Exception: ", e)
            print("Bad Image: ", self.imgs[idx])
            exit()
    
    # Displays a random batch of 64 samples
    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size,shuffle=True)
        
        # get the first batch
        (imgs, labels, idxs) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        obj_classes = list(self.classes)
        
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


class TrashDatasetNumPy(Dataset):
    def __init__(self,img_dir_path,transform,normalize):
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.normalize = normalize
        
        
        # get all training samples/labels by getting paths of the images in each subfolder folder
        self.imgs = np.load(img_dir_path)['x_train']
        self.labels = np.load(img_dir_path)['y_train']
        
    
    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        # load the image
        try:
            img = self.imgs[idx]
            
            # apply any transformation
            if self.transform:
                img = self.transform(img)
                if self.normalize:
                    norm = torchvision.transforms.Normalize((torch.mean(img)),(torch.std(img)))
                    img = norm(img)
                    
            
            # get the label
            label = self.labels[idx]
            
            # return the sample (img (tensor)), object class (int), sample index (int))
            return img, label, idx
        except (ValueError, RuntimeWarning,UserWarning) as e:
            print("Exception: ", e)
            print("Bad Image: ", self.imgs[idx])
            exit()
    
    # Displays a random batch of 64 samples
    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size,shuffle=True)
        
        # get the first batch
        (imgs, labels, idxs) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        obj_classes = ["box","glass","can","crushed can","plastic"]
        
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
        
        
        
'''
Sorting Dataset Class
Parameters:
  img_dir_path - Full path to directory with the images for this dataset.
                 This assumes that the subdirectories contain each class, 
                 only images are in these subdirectories, and that the
                 subdirectory basenames are the desired name of the object class.
                 i.e. dog/dog1.png, cat/cat1.png, etc.

  transform -    Specifies the image format (size, RGB, etc.) and augmentations to use
'''
class SortingDataset(Dataset):
    def __init__(self,img_dir_path,transform,normalize):
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.normalize = normalize
        
        # collect img classes from dir names
        img_classes = next(os.walk(img_dir_path))[1]
        
        # generate a dictionary to map class names to integers idxs
        self.classes = {img_classes[i] : i for i in range(0, len(img_classes))}
        
        # get all training samples/labels by getting paths of the images in each subfolder folder
        self.imgs = []
        self.labels = []
        i = 0
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            if idx > 0: # we don't want the files in the top folder
                for file in path_obj[2]: # path_obj[2] is list of files in the subdirectory
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path
                    self.labels.append(self.classes[os.path.basename(os.path.dirname(self.imgs[i]))]) # get the label from the directory name
                    i+=1
                    
    def read_img(self,img_path):
        w = 128
        h = 128
        c = 3
        img = np.zeros((w,h,c),dtype=np.uint8)
        img_file = open(img_path,"rb")
        
        pixel_h = img_file.read(1)
        pixel_l = img_file.read(1)

        idx = 0
        while pixel_h:
            # extract the RGB values
            r = (pixel_h[0] & 0b11111000)>>3
            g = ((pixel_h[0] & 0b00000111)<<3) | ((pixel_l[0] & 0b11100000)>>5)
            b = pixel_l[0] & 0b00011111
            
            # get the x,y coordinate of the pixel
            x = idx%w
            y = idx//w
            
            # scale to RGB888 and save
            img[x,y,0] = (r<<3)
            img[x,y,1] = (g<<2)
            img[x,y,2] = (b<<3)
            idx += 1
            
            pixel_h = img_file.read(1)
            pixel_l = img_file.read(1)
            
        return img#torch.from_numpy(img)

    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # load the image
        try:
            img = self.read_img(self.imgs[idx])
            
            # apply any transformation
            if self.transform:
                img = self.transform(img)
                if self.normalize:
                    norm = torchvision.transforms.Normalize((torch.mean(img)),(torch.std(img)))
                    img = norm(img)
                    
            
            # get the label
            label = self.labels[idx]
            
            # return the sample (img (tensor)), object class (int), sample index (int))
            return img, label, idx
        except (ValueError, RuntimeWarning,UserWarning) as e:
            print("Exception: ", e)
            print("Bad Image: ", self.imgs[idx])
            exit()
    
    # Displays a random batch of 64 samples
    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size,shuffle=True)
        
        # get the first batch
        (imgs, labels, idxs) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        obj_classes = list(self.classes)
        
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
        
        
class ActivityDataset(Dataset):
    def __init__(self,dataset_path,transform,normalize):
        self.normalize = normalize
        self.transform = transform
        
        # load the csv
        self.activity_df = pd.read_csv(dataset_path)
        
        # collect img classes from dir names
        activity_classes = self.activity_df.Activity.unique()
        
        # generate a dictionary to map class names to integers idxs
        self.classes = {activity_classes[i] : i for i in range(0, len(activity_classes))}
        
    
    def __len__(self):
        return self.activity_df.shape[0]
    
    def __getitem__(self, idx):
        data_vector = self.activity_df.iloc[idx][:561]
        label = self.activity_df.iloc[idx]['Activity']
        
        if self.normalize:
            data_vector = (data_vector-np.mean(data_vector))/np.std(data_vector)
           
        data_vector = torch.from_numpy(data_vector.values.astype(np.float32))
        
        label = torch.tensor(self.classes[label])
        
        return data_vector,label
                    
         
# ===================================================================================================== #
# ====================================== CREATE DATASET FUNCTIONS ===================================== #
# ===================================================================================================== #
def create_MNIST_dataset():
    train_dataset = torchvision.datasets.MNIST('/home/geffen/Desktop/MNIST/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                            #    torchvision.transforms.Normalize(
                            #      (0.1307,), (0.3081,))
                             ]))
    test_dataset = torchvision.datasets.MNIST('/home/geffen/Desktop/MNIST/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                            #    torchvision.transforms.Normalize(
                            #      (0.1307,), (0.3081,))
                             ]))
    #print(train_dataset[0][0].size())
    #exit()
    return train_dataset,test_dataset

# ------------------------------------------------------------------------------------------------ #

def create_ObjectClassifier128_dataset(args):
    normalize = False
    if args.normalize == "y":
        normalize = True
        print("normalize")
    
    # determine the data transformations and format
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
           # transforms.RandomHorizontalFlip(),
            #transforms.RandomAffine(degrees=5,scale=(0.45,1.25),translate=(0.3,0.3))
        ])
    
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    
    img_dir_path = "/home/geffen/Downloads/trash_dataset/binary/"
    
    # create the datasets
    train_dataset = ObjectClassifierDataset(os.path.join(img_dir_path,"train"),train_transform,normalize)
    test_dataset = ObjectClassifierDataset(os.path.join(img_dir_path,"test"),test_transform,normalize)
    
    return train_dataset, test_dataset

def create_Sorting128_dataset(args):
    normalize = False
    if args.normalize == "y":
        normalize = True
        print("normalize")
    
    # determine the data transformations and format
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
           # transforms.RandomHorizontalFlip(),
            #transforms.RandomAffine(degrees=5,scale=(0.45,1.25),translate=(0.3,0.3))
        ])
    
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    
    img_dir_path = "/home/geffen/Desktop/sorting/"
    
    # create the datasets
    train_dataset = SortingDataset(os.path.join(img_dir_path,"train"),train_transform,normalize)
    test_dataset = SortingDataset(os.path.join(img_dir_path,"test"),test_transform,normalize)
    
    return train_dataset, test_dataset

def create_Activity_dataset(normalize=False):
    
    # determine the data transformations and format
    train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    data_dir_path = "/home/geffen/Downloads/archive/"
    
    # create the datasets
    train_dataset = ActivityDataset(os.path.join(data_dir_path,"train.csv"),train_transform,normalize)
    test_dataset = ActivityDataset(os.path.join(data_dir_path,"test.csv"),test_transform,normalize)
    
    return train_dataset, test_dataset

# ===================================================================================================== #
# ============================================= MAIN ================================================== #
# ===================================================================================================== #

if __name__  == "__main__":
    # define training data format
    # test_transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((128, 128)),
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.ToTensor(),
    #     ])
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
           # transforms.RandomHorizontalFlip(),
            #transforms.RandomAffine(degrees=5,scale=(0.45,1.25),translate=(0.3,0.3))
        ])
    # img_data_dir1 = "/home/geffen/Desktop/Face_Detector/assemble_face_dataset_utils/face_classifier_dataset/test"
    # faces_dataset = ObjectClassifierDataset(img_data_dir1,test_transform)
    # faces_dataset.visualize_batch()
    class a:
        def __init__(self,norm):
            self.normalize = norm
    args = a(True)
    
    #dataset = TrashDatasetNumPy("/home/geffen/Downloads/recycle_data_shuffled.npz",train_transform,True)
    dataset = SortingDataset("/home/geffen/Desktop/sorting/train",train_transform,False)
    dataset.visualize_batch()
    # loader = DataLoader(dataset,batch_size=64)
    # for i,batch in enumerate(loader):
    #     print(i)