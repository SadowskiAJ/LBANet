from array import array
import cv2
from scipy.interpolate import griddata
import numpy as np
import sys


### This is a Python 3 script that will not run if called from the ABAQUS Python 2 interpreter directly ###
### Usage: python Py3_Binary_to_Image file_root MaxPixelDim ###
### Can be run independently on an appropriately formatted .bin file, but is called indirectly by APy2_ABQDeformations_to_Binary.py ###

# (C) Dr Adam Jan Sadowski of Imperial College London, last modified at 19.33 on 31/05/22
# Copyright under a BSD 3-Clause License, see https://github.com/SadowskiAJ/LBANet.git


# Open binary file and read from it
file_root = sys.argv[1]
fid = open(file_root + '.bin','rb')

# Read in node count stored as a 4-byte unsigned int
NumVals = array('I', [])
NumVals.fromfile(fid, 1) 
NumVals = int(NumVals[0])

# Read in data ranges
Ranges = array('d', [])
Ranges.fromfile(fid, 5)
minS, maxS, minM, maxM, maxabsW = Ranges[0], Ranges[1], Ranges[2], Ranges[3], Ranges[4]
SRange = maxS - minS # S axis range
MRange = maxM - minM # M axis range
AR = MRange / SRange # Canvas aspect ratio

# Initialise pixel canvas
MaxPixelDim = int(sys.argv[2]) # Max. pixels along the longest dimension
if AR > 1.0: PixelRows, PixelColumns, ScaleFactor = MaxPixelDim, int(np.ceil(MaxPixelDim / AR)), MaxPixelDim / MRange # Slender canvas
else: PixelRows, PixelColumns, ScaleFactor = int(np.ceil(MaxPixelDim * AR)), MaxPixelDim, MaxPixelDim / SRange # Squat canvas
PixelMatrix8 = np.zeros((PixelRows, PixelColumns, 3), np.uint8) # 8-bit colour depth pixel matrix

# Read in transformed geometric buckle data
points, values = [], []
for I in range(NumVals):
	VALS = array('d', [])
	VALS.fromfile(fid, 3) # Read in data each as an 8-byte double - M position, S position, W value
	points.append([(VALS[1] - minM) * ScaleFactor, (VALS[0] - minS) * ScaleFactor])
	values.append(VALS[2])
fid.close()

# Map the geometric coordinates to pixels and write to .jpg file
PixelGridRows, PixelGridCols = np.mgrid[range(0, PixelRows), range(0, PixelColumns)]
PixelGridWs = griddata(np.array(points), np.array(values), (PixelGridRows, PixelGridCols), method='linear', fill_value=0)
ColourDepth8 = 2**8 - 1
for ROW in range(0, PixelRows):
	for COL in range(0, PixelColumns):
		Wnorm = PixelGridWs[ROW, COL] / maxabsW
		R = 0.5 * (1.0 + np.sign(Wnorm)) * abs(Wnorm)
		B = 0.5 * (1.0 - np.sign(Wnorm)) * abs(Wnorm) 		
		PixelMatrix8[PixelRows - ROW - 1, PixelColumns - COL - 1] = (int(np.ceil(B * ColourDepth8)), 0, int(np.ceil(R * ColourDepth8)))

cv2.imwrite(file_root + '.jpg', PixelMatrix8) # OpenCV accepts input in 8-bit BGR format