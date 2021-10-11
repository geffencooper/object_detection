'''
network_train.py
This file builds the dataset, dataloader, NN,
and defines the training loop and stat helper functions
'''

from sklearn.utils import class_weight
from torch.optim import optimizer
from pytorch_dataset import *
from network_def import MNISTClassifier, ObjectClassifier128
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import time
import sys
import argparse
import os
import pandas as pd
import numpy as np
import sklearn.utils as sku
from sklearn.model_selection import KFold, StratifiedKFold
from pytorch_dataset import *

dist = [0,0]
torch.manual_seed(42)
# ===================================================================================================== #
# ====================================== TRAINING FUNCTION ============================================ #
# ===================================================================================================== #
'''
Generic function for training neural networks
Parameters:
    args - these are the args specified in the config file and specify
           the hyperparameters and other training information
'''
def train_nn(args):
    # get the device, hopefully a GPU
    if args.gpu_i >= 0:
        torch.cuda.set_device(int(args.gpu_i))
    device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

    # print important model training info
    print("\n\n\n================================ Start Training ================================")
    print("\nSession Name:",args.session_name)
    print("\nModel Name:",args.model_name)
    if args.gpu_i >= 0:
        print("\nDevice:",torch.cuda.current_device()," ----> ",torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Device:", device)
    print("\nHyperparameters:")

    print("Batch Size: {}\nLearning Rate: {}\nNumber of Epochs: {}\nNormalization:{}\n".format(\
        args.batch_size,args.lr,args.num_epochs,args.normalize))

    # global variables
    best_val_accuracy = 0
    lowest_val_loss = 1e+5
    train_losses = []
    val_losses = []
    val_accuracies = []
    iterations = []
    epochs = []
    curr_train_loss = 0
    
    # track best accuracy from each fold
    fold_accuracies = {}


    # Create and load the datasets
    train_dataset, test_dataset = create_dataset(args)
    
    # prepare for k-fold cross-validation
    #kfold = StratifiedKFold(n_splits=int(args.k_folds), shuffle=True)
    kfold = KFold(n_splits=int(args.k_folds), shuffle=True)


    # get the next set of idxs for the current fold
    #for fold, (train_idxs, val_idxs) in enumerate(kfold.split(train_dataset,train_dataset.labels)):
    for fold, (train_idxs, val_idxs) in enumerate(kfold.split(train_dataset)):
        # create random samplers used by the dataloader
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idxs)
        
        # Define data loaders for training and validation
        train_loader = create_loader(train_dataset,train_subsampler,args)
        val_loader = create_loader(train_dataset,val_subsampler,args)
        
        # Print
        print(f'\nFOLD {fold}')
        print('=================================================================')
        
        
        # build and load a new model
        model,criterion = create_model(args,device)
        torch.save(model.state_dict(),os.path.join(args.log_dest,"_fold_"+str(fold)+"_INIT_model.pth"))
        model.to(device)

        # optimization criteria
        optimizer = create_optimizer(model,args)

        try:
            # track model training time
            start = time.time()

            # track total iterations
            num_iter = 0

            # ----------------------------------------------------- Training Loop -----------------------------------------------------
            for epoch in range(args.num_epochs):
                # get the next batch
                for i, batch in enumerate(train_loader):
                    # load the data and labels to the gpu
                    X_inputs,Y_labels = to_gpu(batch,device,args)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward pass
                    Y_hat_preds = forward_pass(X_inputs,batch,model,args)

                    # backward pass
                    loss = criterion(Y_hat_preds,Y_labels)
                    if torch.isnan(loss):
                        print("********** NAN ERROR ************")
                        print("output:",Y_hat_preds)
                        print("labels:",Y_labels)
                        print("idxs:",get_idxs(batch,args))
                        exit()
                    curr_train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    
                    # print training statistics every n batches
                    if i % args.loss_freq == 0 and i != 0:
                        print("Train Epoch: {} Iteration: {} [{}/{} ({:.0f}%)]\t Batch {} Loss: {:.6f}".format(epoch,i,i*args.batch_size,(len(train_loader)*args.batch_size),100.*i/(len(train_loader)),i,loss.item()))

                    # check if need to so a mid-dataset validation pass
                    if args.val_freq != 0:    
                        # do a validation pass every m batches (may not want to wait till the end of an epoch)
                        if i % args.val_freq == 0 and i != 0:
                            print("\n\n----------------- Epoch {} Iteration {} -----------------\n".format(epoch,i))

                            # keep track of training and validation loss, since training forward pass takes a while do every m iterations instead of every epoch
                            num_iter += args.val_freq
                            iterations.append(num_iter)
                            train_loss = curr_train_loss/(args.val_freq) # average batch training loss over m iterations instead of over the entire dataset
                            curr_train_loss = 0 # reset
                            train_losses.append(train_loss) 

                            # validation pass
                            accuracy,val_loss = eval_model(model,val_loader,device,criterion,args)
                            val_accuracies.append(accuracy)
                            val_losses.append(val_loss)
                            print("Training Loss:{:.4f}".format(train_loss))


                            # save the most accuracte model up to date
                            if args.classification == "y":
                                if accuracy > best_val_accuracy:
                                    best_val_accuracy = accuracy
                                    torch.save(model.state_dict(),os.path.join(args.log_dest,"BEST_model.pth"))
                                print("Best Accuracy: {:.6f}%".format(best_val_accuracy))
                            elif args.regression == "y":
                                # first time so init lowest val loss to first value
                                if args.val_freq == i and epoch == 0:
                                    lowest_val_loss = val_loss
                                elif val_loss < lowest_val_loss:
                                    lowest_val_loss = val_loss
                                    torch.save(model.state_dict(),os.path.join(args.log_dest,"BEST_model.pth"))
                                print("Lowest Validation Loss: {:.6f}".format(lowest_val_loss))


                            # print the time elapsed
                            end = time.time()
                            elapsed = end-start
                            minutes,seconds = divmod(elapsed,60)
                            hours,minutes = divmod(minutes,60)
                            print("Time Elapsed: {}h {}m {}s".format(int(hours),int(minutes),int(seconds)))
                            print("\n--------------------------------------------------------\n\n")
                
                # validation pass at end of the epoch
                if args.val_freq == 0:
                    print("\n\n----------------- Epoch {} -----------------\n".format(epoch))

                    # keep track of training and validation loss
                    epochs.append(epoch)
                    num_batches = len(train_loader)
                    train_loss = curr_train_loss/num_batches
                    train_losses.append(train_loss) # average batch loss
                    curr_train_loss = 0 # reset

                    # validation pass
                    accuracy,val_loss = eval_model(model,val_loader,device,criterion,args)
                    val_accuracies.append(accuracy)
                    val_losses.append(val_loss)
                    print("Training Loss:{:.4f}".format(train_loss))

                    # save the most accuracte model up to date
                    if args.classification == "y":
                        if accuracy > best_val_accuracy:
                            best_val_accuracy = accuracy
                            torch.save(model.state_dict(),os.path.join(args.log_dest,"BEST_model.pth"))
                        print("Best Accuracy: {:.6f}%".format(best_val_accuracy))
                    elif args.regression == "y":
                        # first time so init lowest val loss to first value
                        if args.val_freq == i and epoch == 0:
                            lowest_val_loss = val_loss
                        elif val_loss < lowest_val_loss:
                            lowest_val_loss = val_loss
                            torch.save(model.state_dict(),os.path.join(args.log_dest,"BEST_model.pth"))
                        print("Lowest Validation Loss: {:.6f}".format(lowest_val_loss))

                    # print the time elapsed
                    end = time.time()
                    elapsed = end-start
                    minutes,seconds = divmod(elapsed,60)
                    hours,minutes = divmod(minutes,60)
                    print("Time Elapsed: {}h {}m {}s".format(int(hours),int(minutes),int(seconds)))
                    print("\n--------------------------------------------------------\n\n")

            print("================================ Finished Training ================================")
            torch.save(model.state_dict(),os.path.join(args.log_dest,"END_model.pth"))
            
            # validation pass
            accuracy,val_loss = eval_model(model,val_loader,device,criterion,args)
            
            # save the most accuracte model up to date
            if args.classification == "y":
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    torch.save(model.state_dict(),os.path.join(args.log_dest,"BEST_model.pth"))
                print("Best Accuracy: {:.6f}%".format(best_val_accuracy))
            elif args.regression == "y":
                # first time so init lowest val loss to first value
                if args.val_freq == i and epoch == 0:
                    lowest_val_loss = val_loss
                elif val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    torch.save(model.state_dict(),os.path.join(args.log_dest,"BEST_model.pth"))
                print("Lowest Validation Loss: {:.6f}".format(lowest_val_loss))

            # print the time elapsed
            end = time.time()
            elapsed = end-start
            minutes,seconds = divmod(elapsed,60)
            hours,minutes = divmod(minutes,60)
            print("Time Elapsed: {}h {}m {}s".format(int(hours),int(minutes),int(seconds)))

            if args.val_freq != 0:
                print("Iterations:",iterations)
            else:
                print("Epochs:",epochs)
            if args.classification == "y":
                print("Val_Accuracies:",val_accuracies)
            print("Val_Losses:",val_losses)
            print("Train_Losses:",train_losses)
            
            fold_accuracies[fold] = best_val_accuracy



        except KeyboardInterrupt:
            torch.save(model.state_dict(),os.path.join(args.log_dest,"MID_model.pth"))
            print("================================ QUIT ================================\n Saving Model ...") 
            
            # validation pass
            accuracy,val_loss = eval_model(model,val_loader,device,criterion,args)

            # save the most accuracte model up to date
            if args.classification == "y":
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    torch.save(model.state_dict(),os.path.join(args.log_dest,"BEST_model.pth"))
                print("Best Accuracy: {:.6f}%".format(best_val_accuracy))
            elif args.regression == "y":
                # first time so init lowest val loss to first value
                if args.val_freq == i and epoch == 0:
                    lowest_val_loss = val_loss
                elif val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    torch.save(model.state_dict(),os.path.join(args.log_dest,"BEST_model.pth"))
                print("Lowest Validation Loss: {:.6f}".format(lowest_val_loss))

            # print the time elapsed
            end = time.time()
            elapsed = end-start
            minutes,seconds = divmod(elapsed,60)
            hours,minutes = divmod(minutes,60)
            print("Time Elapsed: {}h {}m {}s".format(int(hours),int(minutes),int(seconds)))

            print("Iterations:",iterations)
            if args.classification == "y":
                print("Val_Accuracies:",val_accuracies)
            print("Val_Losses:",val_losses)
            print("Train_Losses:",train_losses)

    # Print fold results
    print(f'\n\nK-FOLD CROSS VALIDATION RESULTS FOR {int(args.k_folds)} FOLDS')
    print('-----------------------------------------------------')
    sum = 0.0
    for key, value in fold_accuracies.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(fold_accuracies.items())} %')
# ===================================== Model Dependent Functions =====================================

# create the dataset used by the corresponding model
def create_dataset(args):
    if args.model_name == "MNISTClassifier":
        return create_MNIST_dataset()
    elif args.model_name == "ObjectClassifier128":
        return create_ObjectClassifier128_dataset(args)
    else:
        print("ERROR: invalid model name")
        exit(1)

# create the dataloader based on the collate_fn used by the corresponding model
def create_loader(dataset,sampler,args):
    if args.model_name == "MNISTClassifier":
        return DataLoader(dataset,batch_size=args.batch_size,sampler=sampler)
    elif args.model_name == "ObjectClassifier128":
        return DataLoader(dataset,batch_size=args.batch_size,sampler=sampler)
    else:
        print("ERROR: invalid model name")
        exit(1)

# create the optimizer specified
def create_optimizer(model,args):
    if args.optim == "Adam":
        if args.l2_reg == "y":
            return torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay_amnt)
        else:
            return torch.optim.Adam(model.parameters(),lr=args.lr)
    elif args.optim == "SGD":
        if args.l2_reg == "y":
            return torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay_amnt)
        else:
            return torch.optim.SGD(model.parameters(),lr=args.lr)
    elif args.optim == "RMS":
        if args.l2_reg == "y":
            return torch.optim.RMSprop(model.parameters(),lr=args.lr,weight_decay=args.weight_decay_amnt)
        else:
            return torch.optim.RMSprop(model.parameters(),lr=args.lr)
    else:
        print("ERROR: invalid optimizer name")
        exit(1)

# create the model and loss function based on the name specified and classification/regression options
def create_model(args,device):
    if args.model_name == "MNISTClassifier":
        return MNISTClassifier(),torch.nn.CrossEntropyLoss()
    elif args.model_name == "ObjectClassifier128":
        return ObjectClassifier128(args),torch.nn.CrossEntropyLoss()
        # if args.classification == "y":
        #     return ObjectClassifier128(args),torch.nn.CrossEntropyLoss()
        # elif args.regression == "y":
        #     return ObjectClassifier128(args),torch.nn.MSELoss()
    else:
        print("ERROR: invalid model name")
        exit(1)

# load the batch to the gpu, based on the model the batch has a different structure
# return as (data),labels
def to_gpu(batch,device,args):
    if args.model_name == "MNISTClassifier":
        X_inputs,Y_labels = batch[0].to(device),batch[1].to(device)
        return X_inputs,Y_labels
    elif args.model_name == "ObjectClassifier128":
        X_inputs,Y_labels = batch[0].to(device),batch[1].to(device)
        return X_inputs,Y_labels
    else:
        print("ERROR: invalid model name")
        exit(1)

# do a forward pass using the data and batch structure for the corresponding model
def forward_pass(data,batch,model,args):
    if args.model_name == "MNISTClassifier":
        return model(data)
    elif args.model_name == "ObjectClassifier128":
        return model(data)
    else:
        print("ERROR: invalid model name")
        exit(1)

# get the idxs based on the batch structure of the corresponding model
def get_idxs(batch,args):
    if args.model_name == "MNISTClassifier":
        return None
    elif args.model_name == "ObjectClassifier128":
        return batch[2]
    else:
        print("ERROR: invalid model name")
        exit(1)

# get the label distribution in the batch
def get_label_dist(batch,args):
    if args.model_name == "ObjectClassifier128":
        labels = batch[1]
        for l in labels:
            dist[labels[l]]+=1
        print(dist)
    else:
        print("ERROR: invalid model name")
        exit(1)

'''Helper function to evaluate the network (used during training, validation, and testing)'''
def eval_model(model,data_loader,device,criterion,args,print_idxs=False):
    model.eval()
    model.to(device)

    eval_loss = 0
    correct = 0

    all_preds = torch.tensor([])
    all_labels = torch.tensor([],dtype=torch.int)
    all_idxs = torch.tensor([],dtype=torch.int)
    all_preds = all_preds.to(device)
    all_labels = all_labels.to(device)
    all_idxs = all_idxs.to(device)
    num_batches = len(data_loader)
    with torch.no_grad():
        val_start=time.time()
        for i, (batch) in enumerate(data_loader):
            X_inputs,Y_labels = to_gpu(batch,device,args)
            idxs=[]
            if print_idxs == True:
                idxs = get_idxs(batch,args).to(device)
            
            # forward pass
            Y_hat_preds = forward_pass(X_inputs,batch,model,args)

            # accumulate predictions and labels
            all_preds = torch.cat((all_preds,Y_hat_preds),dim=0)
            all_labels = torch.cat((all_labels,Y_labels),dim=0)
            if print_idxs == True:
                all_idxs = torch.cat((all_idxs,idxs),dim=0)

            # sum up the batch loss
            loss = criterion(Y_hat_preds,Y_labels)
            if torch.isnan(loss):
                print("********** NAN ERROR ************")
                print("output:",Y_hat_preds)
                print("labels:",Y_labels)
                print("idxs:",get_idxs(batch,args))
                print("loss",loss)
                exit()
            eval_loss += loss.item()

            # accumulate the classification predictions
            if args.classification == "y":
                # get the prediction
                pred = Y_hat_preds.max(1,keepdim=True)[1]
                correct += pred.eq(Y_labels.view_as(pred)).sum().item()

        # generate a confusion matrix
        print("validation computation time:",(time.time()-val_start)//60," minutes")
        if args.classification == "y":
            gen_conf_mat(all_preds,all_labels,all_idxs,args.num_classes,print_idxs)
            
        # average the loss over the batches
        eval_loss /= num_batches

        if args.classification == "y":
            print("\nValidation Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(eval_loss,correct,(len(data_loader)*args.batch_size),100.*correct/(len(data_loader)*args.batch_size)))
        else:
            print("\nValidation Loss: {:.4f}".format(eval_loss))
            
        # show predictions vs ground truth
        if print_idxs == True:
            print("validation results")
            for i,p in enumerate(all_preds):
                print("idx: {}, P:{:.4f} GT:{}".format(all_idxs[i].item(),all_preds[i].item(),all_labels[i].item()))


    model.train()
    return 100.*correct/(len(data_loader)*args.batch_size),eval_loss


'''Helper function to create a confusion matrix of classification results'''
# this gets called by eval_model with the predictions and labels
def gen_conf_mat(predictions,labels,idxs,num_classes,print_idxs=False):
    # get the prediction from the max output
    preds = predictions.argmax(dim=1)

    # generate label-prediction pairs
    stacked = torch.stack((preds,labels),dim=1)

    # create the confusion matrix
    conf_mat = torch.zeros(num_classes,num_classes,dtype=torch.int64)

    if print_idxs == True:
        incorrect = []
        # fill the confusion matrix
        for i,pair in enumerate(stacked):
            x,y = pair.tolist()
            conf_mat[x,y] = conf_mat[x,y]+1
            if x!=y:
                incorrect.append((idxs[i].item(),"P:"+str(x)+" GT:"+str(y)))
        print("Incorrect Samples:",incorrect)
    else:
        # fill the confusion matrix
        for pair in stacked:
            x,y = pair.tolist()
            conf_mat[x,y] = conf_mat[x,y]+1

    print("Confusion Matrix")
    print(conf_mat)
    for i in range(num_classes):
        print("class {} accuracy: {:.4f}%".format(i,conf_mat[i,i]*100/torch.sum(conf_mat,dim=0)[i]))

def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")
    # logging details
    parser.add_argument("session_name",help="prefix the logging directory with this name",type=str)
    parser.add_argument("log_dest",help="name of directory with logging info (stats, train model, parameters, etc.)",type=str)

    # dataset details
    parser.add_argument("root_dir",help="full path to dataset directory",type=str)
    parser.add_argument("--train_data_dir",help="specify if training data is in another directory within root_dir",type=str)
    parser.add_argument("--test_data_dir",help="specify if test data is in another directory within root_dir",type=str)
    parser.add_argument("train_labels_csv",help="file name of csv with training labels and/or metadata",type=str)
    parser.add_argument("test_labels_csv",help="file name of csv with validation labels and/or metadata",type=str)
    parser.add_argument("k_folds",help="number of partitions for k-fold cross-validation",type=str)

    # training details
    parser.add_argument("gpu_i",help="GPU instance to use (0, 1, 2, etc.)",type=int)
    parser.add_argument("model_name",help="name of the model class (torch.nn.Module)",type=str)
    parser.add_argument("optim",help="name of PyTorch optimizer to use",type=str)
    parser.add_argument("loss_freq",help="print the loss every nth batch",type=int)
    parser.add_argument("val_freq",help="do a validation pass every nth batch (if set to 0, do every epoch)",type=int)

    # hyperparameters
    parser.add_argument("batch_size",help="what size batch to use (32, 64, 128, etc.)",type=int)
    parser.add_argument("lr",help="learning rate",type=float)
    parser.add_argument("classification",help="use the model for classification (y/n)",type=str)
    parser.add_argument("num_classes",help="number of classes for the task (set to -1 for regression)",type=int)
    parser.add_argument("regression",help="use the model for regression (y/n)",type=str)
    parser.add_argument("num_epochs",help="number of times to go through entire training set",type=int)
    parser.add_argument("normalize",help="normalize input features (y/n)",type=str)
    parser.add_argument("weighted_loss",help="weight loss function based on imbalanced classes (y/n), weights calculated from dataset",type=str)
    parser.add_argument("imbalanced_sampler",help="use an imbalanced sampler to rebalance class distribution per batch (y/n)",type=str)
    parser.add_argument("l2_reg",help="do l2 regularization (y/n)",type=str)
    parser.add_argument("weight_decay_amnt",help="weight decay constant for l2 regularization (float)",type=float)
    parser.add_argument("dropout",help="use dropout before fully connected layer (y/n)",type=str)
    parser.add_argument("dropout_prob",help="dropout probability (float)",type=float)

    # extra optional
    parser.add_argument("--load_trained",help="load a pretrained model (y/n)",type=str)
    parser.add_argument("--trained_path",help="local path to trained model",type=str)

    args = parser.parse_args()

    if args.classification == "y" and args.regression == "y":
        parser.error("can only choose either classification or regression, not both")
    if args.classification == "y" and args.num_classes <= 0:
        parser.error("chose classification but --num_classes is invalid, must specify --num_classes")
    if args.regression == "y" and args.num_classes > 0:
        parser.error("chose regression but also specified --num_classes, invalid selection")
    
    print("============================ Raw Args ============================")
    print(args)

    return args



# ===================================== Main =====================================
if __name__ == "__main__":
    args = parse_args()
    train_nn(args)
    