'''
network_def.py
This file defines multiple CNN models for object classification/detection
'''


import torch
#from torch.cuda import set_device
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.utils.rnn as rnn_utils
#import copy 
torch.manual_seed(42)

# ===================================================================================================== #
# ============================================= MODELS ================================================ #
# ===================================================================================================== #
'''
MNIST CNN used to make sure training is working correctly
Parameters:
    None
'''
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# --------------------------------------------------------------------------------- #

'''
Base CNN used for object classification (images 128x128)
Parameters:
    args - these are the args specified in the config file and are passed
           in by the training function in network_train.py
'''
class ObjectClassifier128(torch.nn.Module):
    def __init__(self,args):
        super(ObjectClassifier128,self).__init__()
        self.args = args

        self.device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

        # generic pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 1x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
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
        self.fc2 = nn.Linear(10, args.num_classes, bias=False)

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
        
        # flatten the batch_size x 64x4x4 tensor to be batch_size x (64*4*4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x



# ==================================================================================

'''
Base CNN used for object classification (images 128x128)
Parameters:
    args - these are the args specified in the config file and are passed
           in by the training function in network_train.py
'''
class SortingClassifier128(torch.nn.Module):
    def __init__(self,args):
        super(SortingClassifier128,self).__init__()
        self.args = args

        self.device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

        # generic pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 1x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias=True)
        
        # 8x128x128 --> 16x64x64 (padding by 1 so same dimension)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        
        # 16x64x64 --> 32x32x32 (padding by 1 so increase dimension)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
        
        # 32x32x32 --> 64x16x16 (padding by 2 so increase dimension)
        self.conv7 = nn.Conv2d(32, 64, 3, padding=2, bias=True)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
                
        # 64x16x16 --> 64x8x8 (padding by 1 so same dimension)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1, bias=True)

        # 64x8x8 --> 64x4x4 (passing by 1 so same dimension)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1, bias=True)

        # flatten to fully connected layer
        self.fc1 = nn.Linear(64*4*4, 10, bias=True)
        self.fc2 = nn.Linear(10, args.num_classes, bias=True)

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
        
        # flatten the batch_size x 64x4x4 tensor to be batch_size x (64*4*4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    # ==================================================================================

'''
FCN
'''
class ActivityFCN(torch.nn.Module):
    def __init__(self,args):
        super(ActivityFCN,self).__init__()
        self.args = args

        self.device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")


        # two fully connected layer
        self.fc1 = nn.Linear(784, 32, bias=True)
        self.fc2 = nn.Linear(32, args.num_classes, bias=True)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """Forward prop"""
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x