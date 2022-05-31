import odbAccess
from odbAccess import *
from array import array
import math
import numpy as np
import os


### This is a Python 2 script that must be run via the ABAQUS Python interpreter ###
### Imported by: APy2_Model_Script.py ###

# (C) Dr Adam Jan Sadowski of Imperial College London, last modified at 19.30 on 31/05/22
# Copyright under a BSD 3-Clause License, see https://github.com/SadowskiAJ/LBANet.git


def deformations_to_image(file_root, path_to_Py3, MaxPixelDim, LBA_eigenmodes, LBAStepName, InstanceName, SurfaceNodeSetName):
	print("Working on " + file_root)
	print("Processing .odb output database to .bin binary")

	myOdb = openOdb(file_root + '.odb',readOnly=True)
	LBA = myOdb.steps[LBAStepName] 
	for mode in range(1, LBA_eigenmodes + 1):
		location = LBA.frames[mode].fieldOutputs['COORD']
		translation = LBA.frames[mode].fieldOutputs['U']

		COORDS = location.getSubset(region=myOdb.rootAssembly.instances[InstanceName].nodeSets[SurfaceNodeSetName])	
		US = translation.getSubset(region=myOdb.rootAssembly.instances[InstanceName].nodeSets[SurfaceNodeSetName])
		NumVals = len(COORDS.values)

		# Open binary file and write to it
		fid = open(file_root + '_' + str(mode) + '.bin.','wb')
		array('I', [NumVals]).tofile(fid) # Save node count as 4-byte unsigned int
		array('d', [0.0, 0.0, 0.0, 0.0, 0.0]).tofile(fid) # Make room for five 8-byte doubles giving ranges (to be written last)

		# This analysis assumes a classical cylindrical coordinate system for analysis, with 1 & 2 as the transverse axes and 3 as the vertical axis
		# For a shell of revolution, the relationship to ABAQUS xyz Cartesian coordinate system is 1=z (pos 2), 2=y (pos 1), 3=x (pos 0).
		minS, maxS, minM, maxM, minW, maxW = np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf
		for I in range(len(COORDS.values)):
			C1, C2, C3 = COORDS.values[I].data[2], COORDS.values[I].data[0], COORDS.values[I].data[1]
			U1, U2 = US.values[I].data[2], US.values[I].data[0]
			S = math.sqrt(C1 * C1 + C2 * C2) * math.atan2(C2, C1)
			M = C3
			W = math.sqrt((C1 + U1) * (C1 + U1) + (C2 + U2) * (C2 + U2)) - math.sqrt(C1 * C1 + C2 * C2)
			array('d', [S, M, W]).tofile(fid)
			if S < minS: minS = S
			if S > maxS: maxS = S
			if M < minM: minM = M
			if M > maxM: maxM = M
			if W < minW: minW = W
			if W > maxW: maxW = W

		fid.seek(4, 0) # Move back to the fourth byte position, after the original 4-byte unsigned int
		maxabsW = max(abs(minW), maxW)
		array('d', [minS, maxS, minM, maxM, maxabsW]).tofile(fid) # Write the ranges represented by five 8-byte doubles to file
		fid.close()

		# Call Python 3 processing script via system call (while clearing python path)
		print("Processing .bin binary to .jpg image")
		tmp = os.environ["PYTHONPATH"]
		os.environ["PYTHONPATH"] = ""
		commandS = path_to_Py3 + ' Py3_Binary_to_Image.py ' + file_root + '_' + str(mode) + ' ' + str(MaxPixelDim)
		os.system(commandS)
		os.environ["PYTHONPATH"] = tmp

	myOdb.close()	
	return True