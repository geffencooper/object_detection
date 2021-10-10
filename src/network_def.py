'''
network_def.py
This file defines multiple CNN models for object classification/detection
'''


import torch
#from torch.cuda import set_device
import torch.nn as nn
import torch.nn.Functional as F
#import torch.nn.utils.rnn as rnn_utils
#import copy 

# This is the base CNN used to train an object classifier (images 128x128)
# The convolution layers are taken from this model for object detection
class ObjectClassifier128(torch.nn.Module):
    def __init__(self,args):
        super(ObjectClassifier128,self).__init__()
        self.args = args

        self.device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

        # generic pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 1x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        
        # 8x128x128 --> 16x64x64 (padding by 1 so same dimension)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        
        # 16x64x64 --> 32x32x32 (padding by 1 so increase dimension)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        
        # 32x32x32 --> 64x16x16 (padding by 2 so increase dimension)
        self.conv7 = nn.Conv2d(32, 64, 3, padding=2, bias=False)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
                
        # 64x16x16 --> 64x8x8 (padding by 1 so same dimension)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1, bias=False)

        # 64x8x8 --> 64x4x4 (passing by 1 so same dimension)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1, bias=False)

        # flatten to fully connected layer
        self.fc1 = nn.Linear(64*4*4, 10, bias=False)
        self.fc2 = nn.Linear(10, 2, bias=False)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """Forward prop"""
        
        # 1x128x128 --> 8x128x128 (padding by 1 so same dimension)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # 8x128x128 --> 16x64x64 (padding by 1 so same dimension)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # 16x64x64 --> 32x32x32 (padding by 1 so increase dimension)
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        # 32x32x32 --> 64x16x16 (padding by 2 so increase dimension)
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        
        # 64x16x16 --> 64x8x8 (padding by 1 so same dimension)
        x = self.pool(x)
        x = F.relu(self.conv9(x))
        
        # 64x8x8 --> 64x4x4 (padding by 1 so same dimension)
        x = self.pool(x)
        x = F.relu(self.conv10(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    # initialize the hidden state at the start of each forward pass
    def init_hidden(self,batch_size):
        if self.init == True:
            self.h0 = torch.randn(self.num_layers,batch_size,self.hidden_size)
        else:
            self.h0 = torch.zeros(self.num_layers,batch_size,self.hidden_size)
        self.h0 = self.h0.to(self.device)



# ==================================================================================

