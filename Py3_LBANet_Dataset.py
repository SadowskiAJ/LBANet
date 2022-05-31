# PyTorch imports
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# All other imports
import os
import numpy as np
import random

### Master Dataset definition for the LBANet Pytorch trainer ###
### Imported by: Py3_LBANet_Train.py ###

# (C) Dr Adam Jan Sadowski of Imperial College London, last modified at 23.51 on 31/05/22
# Copyright under a BSD 3-Clause License, see https://github.com/SadowskiAJ/LBANet.git


# Pad image to a 1000x1000 square with black pixels
class SquarePadding:
    def __call__(self, image):
        c, h, w = image.size()
        pad_LHS, pad_TOP, pad_RHS, pad_BOT = 0, 0, 0, 0
        if h > w: # Slender canvas - h is max
            pad_LHS = int((h - w)/2)
            pad_RHS = h - w - pad_LHS
        else: # Squat canvas - w is max
            pad_TOP = int((w - h)/2)
            pad_BOT = w - h - pad_TOP
        padding = (pad_LHS, pad_TOP, pad_RHS, pad_BOT)
        return F.pad(image, padding, 0, 'constant')

# Split the image according to the LHS or RHS half, translate that half to the other half and overwrite original half with zeros
class SplitPad:
    def __call__(self,image):
        choice = random.choice(["LHS","RHS"])
        if choice == "LHS": # Take the LHS half, shift it to the RHS and pad the LHS with black pixels
            newLHS, newRHS = torch.zeros([3,1000,500], dtype=torch.uint8), image[:,:,0:500]
            return torch.cat((newLHS, newRHS), dim=2)   
        if choice == "RHS": # Take the RHS half, shift it to the LHS and pad the RHS with black pixels
            newLHS, newRHS = image[:,:,500:1000], torch.zeros([3,1000,500], dtype=torch.uint8)
            return torch.cat((newLHS, newRHS), dim=2)            

# Resise local feature to a 1/n^2 of original size (n is a positive integer between 1 and 10 inclusive), and place at random n*n grid location
class ResizeAndMove:
    def __call__(self, image):
        n = random.choice([1,2,4,5,10])
        # N.B. n = 1 gives original 1000x1000 pixel image, n = 10 gives 100x100 pixel image
        new_size = int(1000/n)
        image = transforms.Compose([transforms.Resize(size=new_size)])(image)
        i_pos, j_pos = random.randint(0, 1000 - new_size), random.randint(0, 1000 - new_size)
        new_image = torch.zeros([3,1000,1000], dtype=torch.uint8)
        new_image[:,i_pos:(i_pos + new_size),j_pos:(j_pos + new_size)] = image
        return new_image

# Dataset
class LBA_Images_Dataset(Dataset):
    # Instantiate dataset, load image directory structure and category label structure
    def __init__(self, root_dir, data_labels, apply_transform=True):
        self.root_dir = root_dir
        self.image_paths, self.image_labelIDs, self.data_labels = [], [], data_labels
        self.class_counts = []
        for label_ID in range(len(data_labels)):
            image_list = os.listdir(root_dir + '/' + data_labels[label_ID])
            self.image_paths.extend(image_list)
            self.image_labelIDs.extend(label_ID for i in range(len(image_list)))
            self.class_counts.append(len(image_list))
        self.basic_length = len(self.image_paths)            
        self.labelID2label = {i:j for i,j in enumerate(data_labels)}
        self.label2labelID = dict.fromkeys(data_labels)
        self.label2labelID.update((j,i) for i,j in enumerate(self.label2labelID))
        self.apply_transform = apply_transform
        self.enhanced_length, self.weights = self.basic_length, np.ones([1, self.basic_length], dtype=float) / float(self.basic_length)
        if apply_transform: self.enhanced_length, self.weights = self.class_weights()

    # Return length of dataset
    def __len__(self):
        return self.basic_length # Always returns the length of the 'basic' dataset
        
    # Load and return image sample & its known class label from dataset at an index idx
    def __getitem__(self, idx):
        image_labelID = self.image_labelIDs[idx]
        label = self.labelID2label[image_labelID]
        image_path = self.root_dir + '/' + label + '/' + self.image_paths[idx]
        image = read_image(image_path)
        image = self.image_transform(image, label)
        return image, self.label2labelID[label]

    # Transformation pipeline
    def image_transform(self, image, label):
        # Every image is padded to make it square
        image = transforms.Compose([SquarePadding(),])(image)
        if not self.apply_transform: return image

        # Request specific transformations depending on the class
        trans = []
        if label == "CircComp": trans.append("AsIs")
        if label == "CircCompLocal": trans.append(random.choice(["ResizeAndMove","SplitPad"]))
        if label == "CircCompShearCombi": trans.append(random.choice(["AsIs","SplitPad"]))
        if label == "MerCompAxi": trans.append("AsIs")
        if label == "MerCompChequer": trans.append("AsIs")
        if label == "MerCompLocal": trans.append(random.choice(["ResizeAndMove","SplitPad"]))
        if label == "MerCompOtherKoiter": trans.append("AsIs")
        if label == "MerCompShearCombi": trans.append(random.choice(["AsIs","SplitPad"]))
        if label == "ShearLocal": trans.append("ResizeAndMove")
        if label == "ShearTorsion": trans.append("AsIs")
        if label == "ShearTransverse": trans.append(random.choice(["AsIs","SplitPad"]))

        # Request L-R or U-D flips (or nothing) regardless of class
        trans.append("Flip")

        # Apply transforms
        for transform in trans:
            if transform == "AsIs": continue
            if transform == "Flip": image = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])(image)
            if transform == "SplitPad": image = transforms.Compose([SplitPad(),])(image)     
            if transform == "ResizeAndMove": image = transforms.Compose([ResizeAndMove(),])(image)       
        return image

    # Construct class weights
    def class_weights(self):
        weights, LHS, enhanced_length = np.zeros([1, self.basic_length], dtype=float), 0, 0
        for Class in range(len(self.class_counts)):
            if self.data_labels[Class] == "CircComp": weight = 1
            if self.data_labels[Class] == "CircCompLocal": weight = 3
            if self.data_labels[Class] == "CircCompShearCombi": weight = 3
            if self.data_labels[Class] == "MerCompAxi": weight = 1
            if self.data_labels[Class] == "MerCompChequer": weight = 1
            if self.data_labels[Class] == "MerCompLocal": weight = 3                                   
            if self.data_labels[Class] == "MerCompOtherKoiter": weight = 1
            if self.data_labels[Class] == "MerCompShearCombi": weight = 3 
            if self.data_labels[Class] == "ShearLocal": weight = 3
            if self.data_labels[Class] == "ShearTorsion": weight = 1
            if self.data_labels[Class] == "ShearTransverse": weight = 3                                               
            enhanced_length += weight * self.class_counts[Class]
            RHS = LHS + self.class_counts[Class]
            try:
                weights[:,LHS:RHS] = float(1. / len(self.data_labels)) / float(weight * self.class_counts[Class])
            except:
                weights[:,LHS:RHS] = 0.0
            LHS = RHS        
        return enhanced_length, weights
