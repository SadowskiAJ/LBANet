# PyTorch imports
import torch
from torch.utils import data 
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler

# All other imports
import os
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from Py3_LBANet_Dataset import *


### Master LBANet Pytorch trainining script ###
### Can be run directly from cmd as: python Py3LBANet_Train.py ###

# (C) Dr Adam Jan Sadowski of Imperial College London, last modified at 00.21 on 02/06/22
# Copyright under a BSD 3-Clause License, see https://github.com/SadowskiAJ/LBANet.git


##########
# INPUTS #
##########
# Directory paths
datasets_dir_root_path = 'DNN_LBA_datasets' # relative path to root folder containing datasets

# Datasets and labels
datasets = ['D0','D1'] # Which datasets to train on
splitpad_transforms = [True, False] # Whether to potentially apply split-pad transformations on circ. symmetric, non-axisymmetric and non-periodic classes in these datasets (enhances the datasets)
resizemove_transforms = [True, False] # Whether to potentially apply resize-move transformations on 'Local' classes in these datasets (enhances the datasets)
flip_transforms = [True, True] # Whether to apply flip transforms on all classes these datasets (does not enhance the datasets)
data_labels = ['CircComp', 'CircCompLocal', 'CircCompShearCombi', 'MerCompAxi', 'MerCompChequer', 'MerCompLocal', 'MerCompOtherKoiter', 'MerCompShearCombi', 'ShearLocal', 'ShearTorsion', 'ShearTransverse']

# Global training parameters
CNN_arch = 4 # define which architecture will be imported from file AS_LBAnet_ArchX.py
k_folds = 1 # no. of folds for k-fold cross-validation
# N.B. if the above is set to 1, the entire dataset is used for training the final model (with no cross-validation)
batch_size = 32 # how many samples per batch to load
report_loss_every_X_batches = 50 # report on loss function every X batches
num_workers = 0 # no. of subprocesses to be used in data loading (set to zero if complains 'broken pipe')
restart_file = 'LBANet_D0.pt' # Restart analysis (k_folds = 1 only) - give .pt file name as string to restart

# CNN optimiser hyperparameters (input layer always of size 1000x1000 = 10^6 2-channel RB pixels, with unusued G channel dropped entirely)
LR = 0.001 # Learning rate to determine size of optimiser steps
momentum = 0.00 # Momentum in the direction of steepest gradient for the optimiser
epochs = 20 # Max. no of training epochs

# Set device
target_dev = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use CUDA-enabled GPU if present
 

#################
# PRELIMINARIES #
#################
try: os.mkdir("Output")
except: pass

# Simple decimal to string helper function
def dec2str(num, num_points):
    txt = str(int(num)) + "p"
    num -= int(num)
    for p in range(num_points):
        num *= 10
        txt += str(int(num))
        num -= int(num)
    return txt

# Load CNN architecture
if CNN_arch == 1: from Py3_LBANet_Arch1 import *
if CNN_arch == 2: from Py3_LBANet_Arch2 import *
if CNN_arch == 3: from Py3_LBANet_Arch3 import *
if CNN_arch == 4: from Py3_LBANet_Arch4 import *

# Target fi1e name   
fil += "_LR" + dec2str(LR, 4) + "_MM" + dec2str(momentum, 2) + "_K" + str(k_folds) + "_E" + str(epochs) + "_B" + str(batch_size)


############
# TRAINING #
############
# Initialise datasets
if k_folds > 1: 
    nonenhanced_dataset = LBA_Images_Dataset(root_dir=datasets_dir_root_path, datasets=datasets, data_labels=data_labels)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True) 
else:
    nonenhanced_dataset = LBA_Images_Dataset(root_dir=datasets_dir_root_path, datasets=datasets, data_labels=data_labels,\
         splitpad_transforms=splitpad_transforms, resizemove_transforms=resizemove_transforms, flip_transforms=flip_transforms)

def print_dataset_counts(IDs):
    class_counts = {}
    for labelID in sorted(nonenhanced_dataset.labelID2label):
        class_counts[labelID] = (np.array(IDs) == labelID).sum()
        print("Class " + str(labelID) + " or " + nonenhanced_dataset.labelID2label[labelID] + ": " + str(class_counts[labelID]) + " instances (" + str(np.round(100.*float(class_counts[labelID])/float(len(IDs)),2)) + " % of basic total)")
    print("\n")
print("#################################")
print("Dataset " + "".join(datasets) + " basic instances: " + str(nonenhanced_dataset.basic_length) + " of which:")
print_dataset_counts(nonenhanced_dataset.image_labelIDs)
for D in range(len(datasets)):
    L, IE = nonenhanced_dataset.dataset_lengths[D], nonenhanced_dataset.dataset_enhanced[D]
    print("Dataset " + datasets[D] + " has " + str(L) + " instances out of " + str(np.sum(nonenhanced_dataset.dataset_lengths)) + " (" + str(np.round(100.0 * float(L)/float(np.sum(nonenhanced_dataset.dataset_lengths)), 2)) + "%). Enhanced: " + str(IE))
print("\n")

# Define an epoch (complete pass over the training data)
def train_single_epoch(network, optimiser, loss_fn, loader, flush):
    correct_predictions, current_loss = 0, 0.0
    for i, (inputs, labels) in enumerate(loader): # Load batch
        inputs, labels = inputs.to(device=target_dev), labels.to(device=target_dev) # Make sure data is on the right target device
        for param in network.parameters(): param.grad = None # Reset computed partial gradients
        predictions = network(inputs) # Make network prediction ('forward pass') - N.B. 'predictions' here are raw unnormalised scores / class
        loss = loss_fn(predictions, labels) # Compute loss function
        loss.backward() # Compute partial gradients through back-propagation
        optimiser.step() # Update network parameters
        current_loss += loss.item() # Update current loss value
        _, predictions = torch.max(predictions.data, dim=1) # Obtain predicted class labels
        correct_predictions += (predictions == labels).sum().item()
        if (i%report_loss_every_X_batches) == (report_loss_every_X_batches-1):
            print("Loss after training batch " + str(i+1) + " / " + str(np.round(float(loader.dataset.enhanced_length) / float(batch_size), 2))  + ": " + str(np.round(current_loss, 2)))
    if k_folds == 1: pre = 1.0
    else: pre = float(k_folds) / (float(k_folds) - 1.0) # Adjust for basic dataset size (if for k-fold)
    correct_predictionsPC = 100.0 * pre * float(correct_predictions) / float(loader.dataset.enhanced_length)
    if flush: print("Total after epoch. On the fly loss: " + str(np.round(current_loss, 2)) + ", CA = " + str(np.round(correct_predictionsPC, 2)) + " %", end="", flush=True)
    else: print("Total after epoch. On the fly loss: " + str(np.round(current_loss, 2)) + ", CA = " + str(np.round(correct_predictionsPC, 2)) + " %")
    return correct_predictionsPC, current_loss

# Compute classification accuracy on the dataset in current fold
def classification_accuracy_loss(network, loss_fn, loader):
    correct_predictions, current_loss = 0, 0.0
    with torch.no_grad(): # Autograd not needed for inference of trained network
        for i, (inputs, labels) in enumerate(loader): # Load batch
            inputs, labels = inputs.to(device=target_dev), labels.to(device=target_dev) # Make sure data is on the right target device
            predictions = network(inputs) # Make network prediction ('forward pass')
            current_loss += loss_fn(predictions, labels).item() # Compute and accumulate loss function           
            _, predictions = torch.max(predictions.data, dim=1)
            correct_predictions += (predictions == labels).sum().item()
    return correct_predictions, current_loss


# Update file with on-the-fly per-fold per-epoch information
tic0 = time.perf_counter()
try: os.remove("Output/" + fil)
except: pass
def update_file(fil, txt):
    fid = open(file="Output/" + fil, mode="a")
    fid.write(txt + "\n")
    fid.close()

# Save network as .pt file
def save_network(fil, loss):
    LBA_CNN.to(device="cpu")
    torch.save({'epoch': E, \
                'model_state_dict': LBA_CNN.state_dict(), \
                'optimizer_state_dict': MSGD.state_dict(), \
                'loss': loss, \
                'data_labels': data_labels}, "Output/" + fil)
    torch.cuda.empty_cache()
    LBA_CNN.to(device=target_dev) 

# Main training loop
CACCs_train, CACCs_test, Losses_train, Losses_test, best_loss = [], [], [], [], np.inf # Classification Accuracies and losses for each fold
CELossFn = torch.nn.CrossEntropyLoss() ## Cross-entropy loss function for multi-class classification networks
# k-fold stratifed cross-validation on the basic dataset
if k_folds > 1: 
    for fold, (trn_IDs, tst_IDs) in enumerate(skf.split(X=nonenhanced_dataset.image_paths, y=nonenhanced_dataset.image_labelIDs)):
        print("##### Fold " + str(fold+1) + " of " + str(k_folds) + " #####")
        print("Non-enhanced dataset " + "".join(datasets) + " training instances: " + str(len(trn_IDs)) + " of which:")
        print_dataset_counts(np.array(nonenhanced_dataset.image_labelIDs)[trn_IDs])
        print("Non-enhanced dataset " + "".join(datasets) + " test instances: " + str(len(tst_IDs)) + " of which:")
        print_dataset_counts(np.array(nonenhanced_dataset.image_labelIDs)[tst_IDs])
        print("\n")

        # Initialise non-enhanced dataloaders for this fold
        training_loader = DataLoader(dataset=nonenhanced_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(trn_IDs), num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(dataset=nonenhanced_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(tst_IDs), num_workers=num_workers, pin_memory=True)

        # Initialise network for this fold and print diagnostics
        LBA_CNN = LBANet(data_labels, target_dev) # Initialise network
        print(LBA_CNN)
        LBA_CNN.describe()
        LBA_CNN.to(device=target_dev) # Move network to target device 
        MSGD = torch.optim.SGD(params=LBA_CNN.parameters(), lr=LR, momentum=momentum) # Initialise stochastic gradient descent optimiser with momentum

        # Train network for this fold
        LBA_CNN.train(mode=True)
        CA_per_epoch, training_loss_per_epoch, times_per_epoch = [], [], []
        for E in range(epochs):
            print("### Epoch " + str(E+1) + " of " + str(epochs) + " (batch size = " + str(batch_size) + ") ###")
            txt, tic = "", time.perf_counter()
            CAtrnPC, LStrn = train_single_epoch(network=LBA_CNN, optimiser=MSGD, loss_fn=CELossFn, loader=training_loader, flush=True)
            toc = time.perf_counter()        
            CA_per_epoch.append(CAtrnPC)
            training_loss_per_epoch.append(LStrn)
            times_per_epoch.append(toc - tic)
            if LStrn < best_loss: 
                best_loss = LStrn
                txt = "CNN SAVED!"
                save_network("Arch" + str(CNN_arch) + "_CV_" + "".join(datasets) + ".pt", LStrn)       
            update_file(fil + "_CV.txt", str(fold+1) + ", " + str(E+1) + ", " + str(LStrn) + ", " + str(CAtrnPC) + ", " + str(toc - tic) + ", " + txt + " ##########")     
            print("... elapsed time: " + str(np.round(toc - tic, 2)) + " s. " + txt)

        print("++ Average time per epoch: " + str(np.round(np.average(times_per_epoch), 2)) + " s. ++")
        LBA_CNN.train(mode=False)
        LBA_CNN.eval()
        # N.B. the loss and CA evaluated on the fly during training are not the same as those evaluated in what follows

        # Evaluate network for this fold
        tic = time.perf_counter()
        CAtrn, LStrn = classification_accuracy_loss(network=LBA_CNN, loss_fn=CELossFn, loader=training_loader)
        toc = time.perf_counter()  
        CAtrnPC = 100.0*float(CAtrn)/float(len(trn_IDs))
        print("# Training CA = " + str(CAtrn) + " out of " + str(len(trn_IDs)) + " (" + str(np.round(CAtrnPC, 2)) + " %), loss = " + str(np.round(LStrn, 2)))
        update_file(fil + "_CV.txt", str(fold+1) + ", 0, " + str(LStrn) + ", " + str(CAtrnPC) + ", " + str(toc - tic) + ", " + str(len(trn_IDs)) + ", TRAINING")
        
        tic = time.perf_counter()
        CAtst, LStst = classification_accuracy_loss(network=LBA_CNN, loss_fn=CELossFn, loader=test_loader)
        toc = time.perf_counter()  
        CAtstPC = 100.0*float(CAtst)/float(len(tst_IDs))
        print("# Testing CA = " + str(CAtst) + " out of " + str(len(tst_IDs)) + " (" + str(np.round(CAtstPC, 2)) + " %), loss = " + str(np.round(LStst, 2)))
        update_file(fil + "_CV.txt", str(fold+1) + ", 0, " + str(LStst) + ", " + str(CAtstPC) + ", " + str(toc - tic) + ", " + str(len(tst_IDs)) + ", TEST")

        # Save metrics for this fold
        Losses_train.append(LStrn)      
        CACCs_train.append(CAtrnPC)
        Losses_test.append(LStst)  
        CACCs_test.append(CAtstPC)
        torch.cuda.empty_cache()
        print("\n")

# Training on the enhanced dataset
else: 
    # Initialise enhanced dataset dataloader
    training_loader = DataLoader(dataset=nonenhanced_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, \
        sampler=WeightedRandomSampler(weights=torch.DoubleTensor(nonenhanced_dataset.weights)[0], num_samples=nonenhanced_dataset.enhanced_length, replacement=True))

    # Initialise network and print diagnostics
    net_file = "Arch" + str(CNN_arch) + "_FINAL_" + "".join(datasets) + ".pt"
    first_epoch = 0
    if restart_file is not None:
        net_file = restart_file[:-3] + "_RESTART.pt"
        chkpnt = torch.load("Output/" + restart_file)
        LBA_CNN = LBANet(data_labels, target_dev) # Initialise network
        LBA_CNN.load_state_dict(chkpnt["model_state_dict"])
        LBA_CNN.train(mode=True)
        LBA_CNN.describe()
        LBA_CNN.to(device=target_dev) # Move network to target device         
        MSGD = torch.optim.SGD(params=LBA_CNN.parameters(), lr=LR, momentum=momentum) # Initialise stochastic gradient descent optimiser with momentum    
        MSGD.load_state_dict(chkpnt["optimizer_state_dict"])
        #first_epoch = chkpnt["epoch"] + 1
        #best_loss = chkpnt["loss"]
    else:
        LBA_CNN = LBANet(data_labels, target_dev) # Initialise network
        print(LBA_CNN)
        LBA_CNN.describe()
        LBA_CNN.to(device=target_dev) # Move network to target device 
        MSGD = torch.optim.SGD(params=LBA_CNN.parameters(), lr=LR, momentum=momentum) # Initialise stochastic gradient descent optimiser with momentum    

    # Train network
    print("##### TRAINING ON ENHANCED DATASET #####")
    epoch_times, evaluation_times = [], []
    for E in range(first_epoch, epochs): 
        # Train network
        LBA_CNN.train(mode=True)
        print("### Epoch " + str(E+1) + " of " + str(epochs) + " (batch size = " + str(batch_size) + ") ###")
        txt, tic = "", time.perf_counter()
        CAtrnPC, LStrn = train_single_epoch(network=LBA_CNN, optimiser=MSGD, loss_fn=CELossFn, loader=training_loader, flush=False)
        toc = time.perf_counter()   
        epoch_times.append(toc - tic)     
        # N.B. the 'on the fly' loss and CA are overwritten with evaluation values as follows

        # Evaluate network 
        LBA_CNN.train(mode=False)
        LBA_CNN.eval()        
        tic = time.perf_counter()
        CAtrn, LStrn = classification_accuracy_loss(network=LBA_CNN, loss_fn=CELossFn, loader=training_loader)
        toc = time.perf_counter() 
        evaluation_times.append(toc - tic) 
        CAtrnPC = 100.0*float(CAtrn)/float(nonenhanced_dataset.enhanced_length)

        # Save network
        LBA_CNN.train(mode=True)
        if LStrn < best_loss: 
            best_loss = LStrn
            txt = "CNN SAVED!"
            save_network(net_file, LStrn)   
    
        # Reporting
        update_file(fil + "_FINAL.txt", str(E+1) + ", " + str(LStrn) + ", " + str(CAtrnPC) + ", " + \
            str(epoch_times[-1]) + ", " + str(evaluation_times[-1]) + ", " + txt + " ##########")     
        print("Training actual CA = " + str(CAtrn) + " out of " + str(nonenhanced_dataset.enhanced_length) + \
            " (" + str(np.round(CAtrnPC, 2)) + " %), loss = " + str(np.round(LStrn, 2))) 
        print("... elapsed time: " + str(np.round(epoch_times[-1], 2)) + " s. (train)  " + \
            str(np.round(evaluation_times[-1], 2)) + " s. (eval)  " + txt)
    print("++ Average training time per epoch: " + str(np.round(np.average(epoch_times), 2)) + " s. ++")
    print("++ Average evaluation time per epoch: " + str(np.round(np.average(evaluation_times), 2)) + " s. ++")   
    torch.cuda.empty_cache()
    print("\n")    
toc0 = time.perf_counter()


####################
# REPORTING FOR CV #
####################
if k_folds > 1: # Per-fold summary
    print("Per-fold summary at end of " + str(epochs) + " epochs, elapsed time " + str(np.round(toc0 - tic0, 2)) + " s.")
    titles = ['fold', 'CA train (%)', 'Loss train', 'CA test (%)', 'Loss test']
    data = [titles] + list(zip(range(1, k_folds+1), CACCs_train, Losses_train, CACCs_test, Losses_test))
    for i, d in enumerate(data):
        if i == 0: line = '|'.join(str(x).ljust(12) for x in d)
        else: line = '|'.join(str(np.round(x, 2)).ljust(12) for x in d)
        print(line)
        if i == 0:
            print('-' * len(line))
    print("\n")

    # Overall summary
    def print_stats(txt, lst):
        print(txt + ": min = " + str(np.round(np.min(lst), 2)) + ", max = " + str(np.round(np.max(lst), 2)) + \
            ", mean = " + str(np.round(np.average(lst), 2)), ", stderr = " + str(np.round(np.std(lst, ddof=1)/(float(k_folds)**0.5), 2)))    

    print("Training set performance over " + str(epochs) + " epochs:")
    print_stats("Loss", Losses_train)
    print_stats("C.A.", CACCs_train)
    print("\n")

    print("Test set performance over " + str(epochs) + " epochs:")
    print_stats("Loss", Losses_test)
    print_stats("C.A. (%)", CACCs_test)
    print("\n")
