# `Python` scripts
The following scripts interface mainly with the `ABAQUS Python 2` interpreter:
* `APy2_Generate_Training_Set.py` - A `Python 2` script to be used with the `ABAQUS` `Python 2` interpreter to generate the basic dataset (`D0`) of LBA eigenmodes.
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

# `LBANet` trained network in the `.pt` `PyTorch` format
Blah

# Datasets of LBA eigenmodes of cylindrical shells
The datasets may be downloaded under a CC BY 4.0 license via the following links:
* `D0` on 30/05/22 (basic generic dataset n = 13392, AJS): https://doi.org/10.6084/m9.figshare.19953740
* NOT ACTIVE `D1` on 31/05/22 (wind buckling dataset n = 380, AJS): https://academictorrents.com/download/927280b8dd0d67db124982162943bc657ef1098c


Last updated by Dr Adam Jan Sadowski on 01/06/22.
