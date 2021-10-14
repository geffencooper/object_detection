'''
predict.py
This class is used to try out the network
'''

import torch
from network_def import ObjectClassifier128
from pytorch_dataset import ObjectClassifierDataset
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader
from train_network import train_nn
from train_network import parse_args,eval_model,create_dataset,create_loader,create_model
import copy

if __name__ =="__main__":
    args = parse_args()

    device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = create_dataset(args)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size)

    pmr = ObjectClassifier128(args)
    pmr.load_state_dict(torch.load("../models/ObjectClassifier128_Trash-2021-10-13_22-01-17/fold_0_BEST_model.pth"))

    pmr.eval()
    pmr.to(device)

    eval_model(pmr,test_loader,device,torch.nn.CrossEntropyLoss(),args,print_idxs=True)