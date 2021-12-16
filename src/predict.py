'''
predict.py
This class is used to try out the network
'''

import torch
from network_def import ObjectClassifier128,SortingClassifier128,ActivityFCN
from pytorch_dataset import ObjectClassifierDataset
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader
from train_network import train_nn
from train_network import parse_args,eval_model,create_dataset,create_loader,create_model
import copy
import torch.quantization

if __name__ =="__main__":
    args = parse_args()

    device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = create_dataset(args)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size)

    pmr = ActivityFCN(args)
    pmr.load_state_dict(torch.load("/home/geffen/Desktop/object_detection/models/activity-2021-11-20_14-36-46/fold_0_BEST_model.pth"))
    pmr.eval()
    
    pmr.qconfig = torch.quantization.default_qconfig
    print(pmr.qconfig)
    torch.quantization.prepare(pmr, inplace=True)
    # pmr = torch.quantization.quantize_dynamic(
    # pmr, {torch.nn.Linear}, dtype=torch.qint8
    # )
    print(pmr)
    pmr.to(device)

    print("========== Calibration ===========")
    eval_model(pmr,train_loader,device,torch.nn.CrossEntropyLoss(),args,print_idxs=False)
    
    print("======= Results ========")
    torch.quantization.convert(pmr, inplace=True)
    eval_model(pmr,test_loader,device,torch.nn.CrossEntropyLoss(),args,print_idxs=False)
    