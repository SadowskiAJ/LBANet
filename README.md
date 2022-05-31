# `Python` scripts
The following `Python` scripts are included in this repository:
* `APy2_Generate_Training_Set.py` - A `Python 2` script to be used with the `ABAQUS` `Python 2` interpreter to generate the basic dataset (`D0`) of LBA eigenmodes.
* `APy2_ABQDeformations_to_Binary.py` - A `Python 2` script to be called by the above via the `ABAQUS` `Python 2` interpreter that extracts LBA eigenmode deformation data from an `ABAQUS` `.odb` file and writes it in binary format to an intermediate `.bin` file.
* `Py3_Binary_to_Image.py` - A `Python 3` script to be called by the above via an indirect `os.command()` call to read deformation data from the intermediate `.bin` and encode it as a chromatic signature in a `.jpg` image file.

Additionally:
* `APy2_Cylinder_Wind.py` - A `Python 2` script to be used with the `ABAQUS` `Python 2` interpreter to generate models of a cylindrical shells under non-symmetric wind pressures (used to generate dataset `D1`). It relies on `APy2_ABQDeformations_to_Binary.py` and `Py3_Binary_to_Image.py`.
* `Py3_Classify_Signature.py` - A `Python 3` to be called by the above via an indirect `os.command()` call to classify the chromatic signature using `LBANet` via `PyTorch` dynamically during an `ABAQUS` parametric sweep. 

# `LBANet` trained network in the `.pt` `PyTorch` format
Blah

# Datasets
The datasets may be downloaded from Academic Torrents via the following links:
* `D0` on 30/05/22 (basic dataset, AJS): https://academictorrents.com/download/18e628a6f02706345a6472a467cb646d7e010752


Last updated by Dr Adam Jan Sadowski on 31/05/22.
