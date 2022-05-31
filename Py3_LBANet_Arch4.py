import torch
import torch.nn
import numpy as np

### Class definition for CNN Architecture 4 ###
### Imported by: Py3_LBANet_Train.py ###

# (C) Dr Adam Jan Sadowski of Imperial College London, last modified at 19.45 on 31/05/22
# Copyright under a BSD 3-Clause License, see https://github.com/SadowskiAJ/LBANet.git


# CNN architecture hyperparameters
max_pixel_dim = 1000
Conv1_num_feature_maps = 10 # No. of feature maps (output channels) to be learned by the first convolutional layer - also no. of input channels to second convolutional layer
Conv1_kernel_size = 10 # Kernel 'window' size of the first convolutional layer
Conv1_kernel_stride = 1 # Kernel stride of the first convolutional layer
Conv1_max_pool_size = 2 # Square size of 'maxpool' layer associated with the first convolutional layer
Conv2_num_feature_maps = 20 # No. of feture maps (output channels) to be learned by the second convolutional layer 
Conv2_kernel_size = 10 # Kernel 'window' size of the second convolutional layer
Conv2_kernel_stride = 1 # Kernel stride of the second convolutional layer
Conv2_max_pool_size = 2 # Square size of 'maxpool' layer associated with the second convolutional layer
Conv3_num_feature_maps = 30 # No. of feture maps (output channels) to be learned by the third convolutional layer 
Conv3_kernel_size = 10 # Ker3nel 'window' size of the third convolutional layer
Conv3_kernel_stride = 1 # Kernel stride of the third convolutional layer
Conv3_max_pool_size = 2 # Square size of 'maxpool' layer associated with the third convolutional layer
Conv4_num_feature_maps = 50 # No. of feture maps (output channels) to be learned by the fourth convolutional layer 
Conv4_kernel_size = 10 # Kernel 'window' size of the fourth convolutional layer
Conv4_kernel_stride = 1 # Kernel stride of the fourth convolutional layer
Conv4_max_pool_size = 2 # Square size of 'maxpool' layer associated with the fourth convolutional layer
FC1_output_features = 1024 # No. of output features in first fully-connected layer
FC2_output_features = 512 # No. of output features in second fully-connected layer
FC3_output_features = 256 # No. of output features in third fully-connected layer

# Target file name
fil = "Arch4_C1_" + str(Conv1_num_feature_maps) + "_" + str(Conv1_kernel_size) + "_" + str(Conv1_kernel_stride) + "_" + str(Conv1_max_pool_size) + \
    "_C2_" + str(Conv2_num_feature_maps) + "_" + str(Conv2_kernel_size) + "_" + str(Conv2_kernel_stride) + "_" + str(Conv2_max_pool_size) + \
    "_C3_" + str(Conv3_num_feature_maps) + "_" + str(Conv3_kernel_size) + "_" + str(Conv3_kernel_stride) + "_" + str(Conv3_max_pool_size) + \
    "_C4_" + str(Conv4_num_feature_maps) + "_" + str(Conv4_kernel_size) + "_" + str(Conv4_kernel_stride) + "_" + str(Conv4_max_pool_size)

class LBANet(torch.nn.Module):
    # Instantiate network architecture  
    def __init__(self, data_labels, target_dev):
        super(LBANet, self).__init__()
        self.data_labels, self.target_dev = data_labels, target_dev

        # First convolutional layer of rectified linear units with max-pooling
        self.Conv1 = torch.nn.Conv2d(in_channels=2, out_channels=Conv1_num_feature_maps, kernel_size=Conv1_kernel_size, stride=Conv1_kernel_stride)
        self.ActivConv1 = torch.nn.ReLU()
        self.MaxP1 = torch.nn.MaxPool2d(Conv1_max_pool_size, Conv1_max_pool_size)
        self.C1_output_image_dim = int((max_pixel_dim - (Conv1_kernel_size - 1))/Conv1_max_pool_size)        
        
        # Second convolutional layer of rectified linear units with max-pooling
        self.Conv2 = torch.nn.Conv2d(in_channels=Conv1_num_feature_maps, out_channels=Conv2_num_feature_maps, kernel_size=Conv2_kernel_size, stride=Conv2_kernel_stride)
        self.ActivConv2 = torch.nn.ReLU()
        self.MaxP2 = torch.nn.MaxPool2d(Conv2_max_pool_size, Conv2_max_pool_size)
        self.C2_output_image_dim = int((self.C1_output_image_dim - (Conv2_kernel_size - 1))/Conv2_max_pool_size)

        # Third convolutional layer of rectified linear units with max-pooling
        self.Conv3 = torch.nn.Conv2d(in_channels=Conv2_num_feature_maps, out_channels=Conv3_num_feature_maps, kernel_size=Conv3_kernel_size, stride=Conv3_kernel_stride)
        self.ActivConv3 = torch.nn.ReLU()
        self.MaxP3 = torch.nn.MaxPool2d(Conv3_max_pool_size, Conv3_max_pool_size)
        self.C3_output_image_dim = int((self.C2_output_image_dim - (Conv3_kernel_size - 1))/Conv3_max_pool_size)

        # Fourth convolutional layer of rectified linear units with max-pooling
        self.Conv4 = torch.nn.Conv2d(in_channels=Conv3_num_feature_maps, out_channels=Conv4_num_feature_maps, kernel_size=Conv4_kernel_size, stride=Conv4_kernel_stride)
        self.ActivConv4 = torch.nn.ReLU()
        self.MaxP4 = torch.nn.MaxPool2d(Conv4_max_pool_size, Conv4_max_pool_size)
        self.C4_output_image_dim = int((self.C3_output_image_dim - (Conv4_kernel_size - 1))/Conv4_max_pool_size)

        # First fully-connected layer of rectified linear units
        self.FlattenFC1 = torch.nn.Flatten()
        self.FC1 = torch.nn.Linear(in_features=Conv4_num_feature_maps * self.C4_output_image_dim**2, out_features=FC1_output_features)
        self.ActivFC1 = torch.nn.ReLU()

        # Second fully-connected layer of rectified linear units
        self.FC2 = torch.nn.Linear(in_features=FC1_output_features, out_features=FC2_output_features)
        self.ActivFC2 = torch.nn.ReLU()        

        # Third fully-connected layer of rectified linear units
        self.FC3 = torch.nn.Linear(in_features=FC2_output_features, out_features=FC3_output_features)
        self.ActivFC3 = torch.nn.ReLU()   

        # Fourth fully-connected layer of rectified linear units
        self.FC4 = torch.nn.Linear(in_features=FC3_output_features, out_features=len(self.data_labels))
        self.ActivFC4 = torch.nn.ReLU()           

    # Feedforward batch of instances through network architecture
    def forward(self, x):
        # Stripping out the unused 'G' channel (dim = 1 due to batch mode) and casting uint8 to float32
        x = torch.index_select(x, 1, torch.tensor([0, 2], device=self.target_dev)).float()

        # Passing through first convolutional layer
        x = self.Conv1(x)
        x = self.ActivConv1(x)
        x = self.MaxP1(x)
        
        # Passing through second convolutional layer
        x = self.Conv2(x)
        x = self.ActivConv2(x)
        x = self.MaxP2(x)

        # Passing through third convolutional layer
        x = self.Conv3(x)
        x = self.ActivConv3(x)
        x = self.MaxP3(x)

        # Passing through fourth convolutional layer
        x = self.Conv4(x)
        x = self.ActivConv4(x)
        x = self.MaxP4(x)

        # Passing through first fully-connected layer
        x = self.FlattenFC1(x)
        x = self.FC1(x)
        x = self.ActivFC1(x)

        # Passing through second fully-connected layer
        x = self.FC2(x)
        x = self.ActivFC2(x)        

        # Passing through third fully-connected layer
        x = self.FC3(x)
        x = self.ActivFC3(x)      

        # Passing through fourth fully-connected layer
        x = self.FC4(x)
        x = self.ActivFC4(x)                   
        return x

    # Self-description
    def describe(self):
        print("\nInput layer size:\n 2 x " + str(max_pixel_dim) + " x " + str(max_pixel_dim))
        print("Convolutional ReLU layer 1:\n 2 input channels, " + str(Conv1_kernel_size) + " x " + str(Conv1_kernel_size) + " kernel of stride " + str(Conv1_kernel_stride) + ", " + \
            str(Conv1_num_feature_maps) + " output feature maps of size " + str(self.C1_output_image_dim) + " x " + str(self.C1_output_image_dim))
        print(" (total flat features: " + str(Conv1_num_feature_maps * self.C1_output_image_dim**2) + ")")
        print("Convolutional ReLU layer 2:\n " + str(Conv1_num_feature_maps) + " input channels, " + str(Conv2_kernel_size) + " x " + str(Conv2_kernel_size) + " kernel of stride " + str(Conv2_kernel_stride) + ", " + \
                str(Conv2_num_feature_maps) + " output feature maps of size " + str(self.C2_output_image_dim) + " x " + str(self.C2_output_image_dim))
        print(" (total flat features: " + str(Conv2_num_feature_maps * self.C2_output_image_dim**2) + ")") 
        print("Convolutional ReLU layer 3:\n " + str(Conv2_num_feature_maps) + " input channels, " + str(Conv3_kernel_size) + " x " + str(Conv3_kernel_size) + " kernel of stride " + str(Conv3_kernel_stride) + ", " + \
                str(Conv3_num_feature_maps) + " output feature maps of size " + str(self.C3_output_image_dim) + " x " + str(self.C3_output_image_dim))
        print(" (total flat features: " + str(Conv3_num_feature_maps * self.C3_output_image_dim**2) + ")")       
        print("Convolutional ReLU layer 4:\n " + str(Conv3_num_feature_maps) + " input channels, " + str(Conv4_kernel_size) + " x " + str(Conv4_kernel_size) + " kernel of stride " + str(Conv4_kernel_stride) + ", " + \
                str(Conv4_num_feature_maps) + " output feature maps of size " + str(self.C4_output_image_dim) + " x " + str(self.C4_output_image_dim))
        print(" (total flat features: " + str(Conv4_num_feature_maps * self.C4_output_image_dim**2) + ")")                
        print("Fully-connected linear ReLU layer 1:")  
        print(" " + str(Conv4_num_feature_maps * self.C4_output_image_dim**2) + " input features, " + str(FC1_output_features) + " output features")
        print("Fully-connected linear ReLU layer 2:")  
        print(" " + str(FC1_output_features) + " input features, " + str(FC2_output_features) + " output features")
        print("Fully-connected linear ReLU layer 3:")  
        print(" " + str(FC2_output_features) + " input features, " + str(FC3_output_features) + " output features")        
        print("Fully-connected linear ReLU layer 4:")  
        print(" " + str(FC3_output_features) + " input features, " + str(len(self.data_labels)) + " output features")                
        print("\n")

        floats = 0
        print("Model's state_dict:")
        for param_tensor in self.state_dict(): 
            floats += torch.prod(torch.from_numpy(np.array(self.state_dict()[param_tensor].size())))
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())
        print("\n")
        floats = float(floats)
        print("Total float32s in state dictionary: " + str(int(floats)))
        floats *= 4 # Assuming default float type is float32, so 4 bytes
        print("State dictionary occupies approximately:")
        print(str(floats) + " B")
        floats /= 1e3
        print(str(floats) + " kB")
        floats /= 1e3
        print(str(floats) + " MB")
        floats /= 1e3
        print(str(floats) + " GB\n")        