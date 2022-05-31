# PyTorch imports
import torch
from torchvision.io import read_image

# All other imports
import sys
from Py3_LBANet_Dataset import * # For the image transformations
from Py3_LBANet_Arch4 import * # For the LBANet V4 architecture


### Individual chromatic signature classification using LBANet ###
### This is a Python 3 script that will not run if called from the ABAQUS Python 2 interpreter directly ###
### Usage: python Py3_Classify_Signature file_root LBA_eigenmodes ###
### Can be run independently on an appropriately formatted .bin file, but is called indirectly by APy2_Cylinder_Wind.py ###

# (C) Dr Adam Jan Sadowski of Imperial College London, last modified at 19.37 on 31/05/22
# Copyright under a BSD 3-Clause License, see https://github.com/SadowskiAJ/LBANet.git


# Set device
target_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use CUDA-enabled GPU if present

# Specify network and architecture
network_path = "LBANet.pt" # define which Arch4 network will be used for inference

# Passed as command-line arguments
file_root = sys.argv[1]
LBA_eigenmodes = int(sys.argv[2])

# Load network for inference only
chkpnt = torch.load(network_path) # Load saved checkpoint from .pt file
data_labels = chkpnt["data_labels"]
LBA_CNN = LBANet(data_labels, target_dev)
LBA_CNN.load_state_dict(chkpnt["model_state_dict"])
LBA_CNN.train(mode=False)
LBA_CNN.eval()
LBA_CNN.to(device=target_dev)

for E in range(1,LBA_eigenmodes+1):
	# Classify chromatic signature
	image = read_image(file_root + "_" + str(E) + ".jpg") # Returns a 3xHxW tensor
	image = image_transforms(image) # Pads to a 1000x1000 image
	image = image.unsqueeze(0) # Convert 3D tensor to 4D
	image = image.to(device=target_dev) # Transfer to target device
	prediction = LBA_CNN(image) # Make prediction

	# Manual application of a softmax layer to obtain 'probabilities'
	ps = 100.0 * np.exp(prediction.cpu().detach().numpy()) / np.sum(np.exp(prediction.cpu().detach().numpy()))

	# Write classification to a .txt file
	fid = open(file_root + "_" + str(E) + ".txt",'a')
	fid.write(data_labels[torch.argmax(prediction, dim=1).item()] + "\n")
	fid.write("\n")
	for L in range(len(data_labels)): fid.write(data_labels[L] + ": " + str(np.round(ps[:,L].item(),4)) + " %\n")
	fid.write
	fid.close()