'''
dataset_split.py
This file splits a dataset into training
and test folders. The training data gets
split into validation partitions during
training.
'''

import os
from numpy import tracemalloc_domain
from sklearn.model_selection import train_test_split
import shutil

'''
This function will split a folder which contains
subfolders of classes into train and test folders
with corresponding class subfolders.
Parameters:
  folder_path - the path to the folder with the data
  
  train_percent - the desired percent of the data used for training.
                Test data will be (1 - train_percent)
'''
def split_dataset(folder_path,test_percent):
    # collect img classes from dir names
    img_classes = next(os.walk(folder_path))[1]
    
    # create the train and test folders
    os.mkdir(os.path.join(folder_path,"train"))
    os.mkdir(os.path.join(folder_path,"test"))
    
    # get partitions of all the classes for train and test
    for img_class in img_classes:
        train_files, test_files = split_folder(os.path.join(folder_path,img_class), test_percent)
        
        # move training files to training directory
        os.mkdir(os.path.join(folder_path,"train",img_class))
        for img_file in train_files:
            shutil.move(os.path.join(folder_path,img_class,img_file),os.path.join(folder_path,"train",img_class))
        
        # move test files to test directory
        os.mkdir(os.path.join(folder_path,"test",img_class))
        for img_file in test_files:
            shutil.move(os.path.join(folder_path,img_class,img_file),os.path.join(folder_path,"test",img_class))
            
        # delete the old folder
        os.rmdir(os.path.join(folder_path,img_class))
        

'''
This function will split a folder into training
and test data given a training percent. This assumes
the folder contains a single class of data.
Parameters:
  folder_path - the path to the folder with the images
  
  train_percent - the desired percent of the data used for training.
                Test data will be (1 - train_percent)
'''
def split_folder(folder_path,train_percent):
    # get the file names
    imgs = os.listdir(folder_path)
    train,test = train_test_split(imgs,test_size=train_percent)
    return train,test
    
    
    
if __name__ == "__main__":
    folder_path = "/home/geffen/Downloads/trash_dataset/trash_dataset"
    split_dataset(folder_path,0.15)