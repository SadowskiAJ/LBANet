# `Python` scripts
The following scripts interface mainly with the `ABAQUS Python 2` interpreter:
* `APy2_Generate_Training_Set.py` - A `Python 2` script to be used with the `ABAQUS` `Python 2` interpreter to generate the basic dataset `D0` of LBA eigenmodes.
* `APy2_ABQDeformations_to_Binary.py` - A `Python 2` script to be called by the above via the `ABAQUS` `Python 2` interpreter that extracts LBA eigenmode deformation data from an `ABAQUS` `.odb` file and writes it in binary format to an intermediate `.bin` file.
* `Py3_Binary_to_Image.py` - A `Python 3` script to be called by the above via an indirect `os.command()` call to read deformation data from the intermediate `.bin` and encode it as a chromatic signature in a `.jpg` image file.

Additionally:
* `APy2_Cylinder_Wind.py` - A `Python 2` script to be used with the `ABAQUS` `Python 2` interpreter to generate models of a cylindrical shells under non-symmetric wind pressures (used to generate dataset `D1`). It relies on `APy2_ABQDeformations_to_Binary.py` and `Py3_Binary_to_Image.py`.
* `Py3_Classify_Signature.py` - A `Python 3` to be called by the above via an indirect `os.command()` call to classify the chromatic signature using `LBANet` via `PyTorch` dynamically during an `ABAQUS` parametric sweep. 

The following scripts interface with the `PyTorch` framework via a `Python 3` interpreter:
* `Py3_LBANet_Train.py` - The main script to train the `LBANet` convolutional neural network (CNN) with the `PyTorch` framework and also perform stratified cross-validation.
* `Py3_LBANet_Dataset.py` - A script called by the above to define the `Dataset` object and custom image transformations.
* `Py3_LBANet_Arch1.py` - A script potentially called by the above to define a CNN architecture with one convolutional-maxpool layer pair and three fully-connected layers.
* `Py3_LBANet_Arch2.py` - A script potentially called by the above to define a CNN architecture with two convolutional-maxpool layer pair and three fully-connected layers.
* `Py3_LBANet_Arch3.py` - A script potentially called by the above to define a CNN architecture with three convolutional-maxpool layer pair and four fully-connected layers.
* `Py3_LBANet_Arch4.py` - A script potentially called by the above to define a CNN architecture with four convolutional-maxpool layer pair and four fully-connected layers. This is the architecture ultimately adopted for `LBANet`.

All caused is made available under a BSD 3-Clause license. Please see the license file for details.

# Datasets of LBA eigenmodes of cylindrical shells
The datasets may be downloaded under a CC BY 4.0 license via the following links:
* `D0` on 30/05/22 (basic generic dataset `n = 13,392`): https://figshare.com/s/4697e2d992b7b90f4305
* `D1` on 31/05/22 (wind buckling dataset `n = 380`): https://figshare.com/s/1a73da31fce9b9a29dd3

# `LBANet` trained network in the `.pt` `PyTorch` format
Two trained `LBANet` networks are offered for download from:https://figshare.com/s/9cec97bdbfe0021c44d0
* One trained on the `D0` basic dataset (probabilistically enhanced to `n = 25,726`). Max classification accuracy achieved during training was 99.74%, and that on the non-enhanced `D0` and `D1` are 99.44% and 44.21% respectively, overall 97.92%.
* One trained on both the enhanced `D0` and non-enhanced `D1` datasets (`n = 26,106`). Max classification accuracy achieved during traning was 99.75%, and that on the non-enhanced `D0` and `D1` are 99.81% and 95.26% respectively, overall 99.69%.

Please note that these links will change once the work has been accepted for publication.

Last updated by Dr Adam Jan Sadowski on 15/06/22.
