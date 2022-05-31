# Preliminaries
import math
import os
import shutil
import random
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from odbAccess import *
from APy2_ABQDeformations_to_Binary import *
executeOnCaeStartup()
session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)


### Control script for LBA buckling eigenmode mode training data generation ###
### Can be run directly from cmd as: abaqus cae noGUI=APY2_Generate_Training_Set.py ###

# (C) Dr Adam Jan Sadowski of Imperial College London, last modified at 19.20 on 31/05/22
# Copyright under a BSD 3-Clause License, see https://github.com/SadowskiAJ/LBANet.git


# No. repetitions
nSamples = 50

# Geometry
T = 1.0 # Thickness [L, float]
ROTmin, ROTmax = 100, 1000 # min & max R/t ratio [-, float]
OMEGAmin, OMEGAmax = 0.5, 1.0 # Target dimensionless omega parameter [-, float]

# First run: r/t 10 to 1000, O = 0.1 to 1, O = 1
# Second run: r/t 50 to 200, O = 0.45 to 0.65

# Materials
E = 2e5 # Young's modulus [F/L2, float]
nu = 0.3 # Poisson ratio [-, float]

# Analysis
Analysis, LBA_eigenmodes = 'LBA', 2 # To be reported, LBA always set to compute 10 eigenmodes
path_to_Py3 = """C:\\Users\\"Adam Jan Sadowski"\\AppData\\Local\\Programs\\Python\\Python37\\python.exe"""
abq_command = "abq2017"
MaxPixelDim = 1000

#########################################################################
#########################################################################
#########################################################################
### Do not modify the code unless you know exactly what you are doing ###
#########################################################################
#########################################################################
#########################################################################
# Simple decimal to string helper function
def float2str(flt,decpts):
	strng = str(int(flt)) + "p"
	decpt = 1
	while decpt <= decpts:
		flt -= int(flt)
		flt *= 10
		strng += str(int(flt))
		decpt += 1
	return strng

for repeat in range(nSamples):
	ROT = ROTmin + random.random() * (ROTmax - ROTmin)
	OMEGA = OMEGAmin + random.random() * (OMEGAmax - OMEGAmin)

	R = ROT * T # Radius [L, float]
	L = OMEGA * R * math.sqrt(ROT) # Length [L, float]
	cpus = 1
	meshing, order, el_div = 'QUAD', 1, 100
	Mdb()

	dirs = [Analysis + "_Repository", Analysis + "_Images"]
	for dir in dirs:
		try: 
			os.mkdir(dir)
			if dir != Analysis + "_Repository":
				os.mkdir(dir + "/UM")
				os.mkdir(dir + "/UC")
				os.mkdir(dir + "/UT")
				os.mkdir(dir + "/TS")
				os.mkdir(dir + "/BM")
		except:
			pass

	# ##################### #
	# ### PRELIMINARIES ### #
	# ##################### #
	LBA_REPOSITORY = {'UM' : 1.0, 'UC' : 1.0, 'UT' : 1.0, 'TS' : 1.0, 'BM' : 1.0}

	PROCEDURES = [{'UM' : True,  'UC' : False, 'UT' : False, 'TS' : False, 'BM' : False},
				{'UM' : False, 'UC' : True,  'UT' : False, 'TS' : False, 'BM' : False},
				{'UM' : False, 'UC' : False, 'UT' : True,  'TS' : False, 'BM' : False},
				{'UM' : False, 'UC' : False, 'UT' : False, 'TS' : True,  'BM' : False},
				{'UM' : False, 'UC' : False, 'UT' : False, 'TS' : False, 'BM' : True }, # 5C1

				{'UM' : True,  'UC' : True,  'UT' : False, 'TS' : False, 'BM' : False},			
				{'UM' : True,  'UC' : False, 'UT' : True,  'TS' : False, 'BM' : False},	
				{'UM' : True,  'UC' : False, 'UT' : False, 'TS' : True,  'BM' : False},					
				{'UM' : True,  'UC' : False, 'UT' : False, 'TS' : False, 'BM' : True },	
				{'UM' : False, 'UC' : True,  'UT' : True,  'TS' : False, 'BM' : False},		
				{'UM' : False, 'UC' : True,  'UT' : False, 'TS' : True,  'BM' : False},	
				{'UM' : False, 'UC' : True,  'UT' : False, 'TS' : False, 'BM' : True },		
				{'UM' : False, 'UC' : False, 'UT' : True,  'TS' : True,  'BM' : False},	
				{'UM' : False, 'UC' : False, 'UT' : True,  'TS' : False, 'BM' : True },	
				{'UM' : False, 'UC' : False, 'UT' : False, 'TS' : True,  'BM' : True }, # 5C2
				
				{'UM' : True,  'UC' : True,  'UT' : True,  'TS' : False, 'BM' : False},
				{'UM' : True,  'UC' : True,  'UT' : False, 'TS' : True,  'BM' : False}, 
				{'UM' : True,  'UC' : True,  'UT' : False, 'TS' : False, 'BM' : True }, 
				{'UM' : True,  'UC' : False, 'UT' : True,  'TS' : True,  'BM' : False},	
				{'UM' : True,  'UC' : False, 'UT' : True,  'TS' : False, 'BM' : True },	
				{'UM' : True,  'UC' : False, 'UT' : False, 'TS' : True,  'BM' : True },	
				{'UM' : False, 'UC' : True,  'UT' : True,  'TS' : True,  'BM' : False},			
				{'UM' : False, 'UC' : True,  'UT' : True,  'TS' : False, 'BM' : True },										
				{'UM' : False, 'UC' : True,  'UT' : False, 'TS' : True,  'BM' : True },
				{'UM' : False, 'UC' : False, 'UT' : True,  'TS' : True,  'BM' : True },	# 5C3

				{'UM' : True,  'UC' : True,  'UT' : True,  'TS' : True,  'BM' : False},
				{'UM' : True,  'UC' : True,  'UT' : True,  'TS' : False, 'BM' : True }, 
				{'UM' : True,  'UC' : True,  'UT' : False, 'TS' : True,  'BM' : True }, 
				{'UM' : True,  'UC' : False, 'UT' : True,  'TS' : True,  'BM' : True },
				{'UM' : False, 'UC' : True,  'UT' : True,  'TS' : True,  'BM' : True }, # 5C4

				{'UM' : True,  'UC' : True,  'UT' : True,  'TS' : True,  'BM' : True }]	# 5C5		
				# 5C1 + 5C2 + 5C3 + 5C4 + 5C5 = 5 + 10 + 10 + 5 + 1 = 31 in total	
	
	# ################ #
	# ### GEOMETRY ### #
	# ################ #
	session.viewports['Viewport: 1'].setValues(displayedObject=None)
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=L)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=STANDALONE)
	s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
	s.FixedConstraint(entity=g[2])
	s.Line(point1=(R, 0.0), point2=(R, L))
	s.VerticalConstraint(entity=g[3], addUndoState=False)

	p = mdb.models['Model-1'].Part(name='Shell', dimensionality=THREE_D, type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts['Shell']
	p.BaseShellRevolve(sketch=s, angle=360.0, flipRevolveDirection=OFF)
	s.unsetPrimaryObject()
	session.viewports['Viewport: 1'].setValues(displayedObject=p)
	del mdb.models['Model-1'].sketches['__profile__']

	# ################ #
	# ### MATERIAL ### #
	# ################ #
	mdb.models['Model-1'].Material(name='Generic', description='Generic steel metal.')
	mdb.models['Model-1'].materials['Generic'].Elastic(table=((E, nu), ))

	# ############### #
	# ### SECTION ### #
	# ############### #
	mdb.models['Model-1'].HomogeneousShellSection(name='Shell_Section', 
		preIntegrate=OFF, material='Generic', thicknessType=UNIFORM, thickness=T, 
		thicknessField='', nodalThicknessField='', idealization=NO_IDEALIZATION, 
		poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT, 
		useDensity=OFF, integrationRule=SIMPSON, numIntPts=5)
	p = mdb.models['Model-1'].parts['Shell']
	faces = p.faces.findAt(((0.0, 0.5 * L, R), ))
	region = p.Set(faces=faces, name='Global_Surface_Set')
	p.SectionAssignment(region=region, sectionName='Shell_Section', offset=0.0, 
		offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

	# ################ #
	# ### ASSEMBLY ### #
	# ################ #	
	a = mdb.models['Model-1'].rootAssembly
	a.Instance(name='Shell-1', part=p, dependent=ON)
	f = a.instances['Shell-1'].faces
	faces = f.findAt(((0.0, 0.5 * L, R), ))
	a.Set(faces=faces, name='Global_Surface_Set')
	e = a.instances['Shell-1'].edges
	edges = e.findAt(((0.0, 0.0, R), ))
	a.Set(edges=edges, name='Bot_Edge')
	edges = e.findAt(((0.0, L, R), ))
	a.Set(edges=edges, name='Top_Edge')
	a.DatumCsysByThreePoints(coordSysType=CYLINDRICAL, origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0), point2=(0.0, 0.0, -1.0)) 
	mdb.models['Model-1'].rootAssembly.features.changeKey(fromName='Datum csys-1', toName='Cylindrical_CSYS')
	side1Faces1 = f.findAt(((0.0, 0.5 * L, R), ))
	a.Surface(side1Faces=side1Faces1, name='Global_Surface')

	# ################### #
	# ### INTERACTION ### #
	# ################### #
	REF_BOT = a.ReferencePoint(point=a.instances['Shell-1'].InterestingPoint(edge=e.findAt(coordinates=(0.0, 0.0, R)), rule=CENTER))
	refPoints=(a.referencePoints[REF_BOT.id], )
	a.Set(referencePoints=refPoints, name='BOT_RP')
	REF_TOP = a.ReferencePoint(point=a.instances['Shell-1'].InterestingPoint(edge=e.findAt(coordinates=(0.0, L, R)), rule=CENTER))
	refPoints=(a.referencePoints[REF_TOP.id], )
	a.Set(referencePoints=refPoints, name='TOP_RP')

	mdb.models['Model-1'].MultipointConstraint(name='BOT_Constraint', controlPoint=a.sets['BOT_RP'], surface=a.sets['Bot_Edge'], mpcType=BEAM_MPC, 
		userMode=DOF_MODE_MPC, userType=0, csys=None)
	mdb.models['Model-1'].MultipointConstraint(name='TOP_Constraint', controlPoint=a.sets['TOP_RP'], surface=a.sets['Top_Edge'], mpcType=BEAM_MPC, 
		userMode=DOF_MODE_MPC, userType=0, csys=None)

	# ########### #
	# ### BCs ### #
	# ########### #
	mdb.models['Model-1'].EncastreBC(name='BOT_BC', createStepName='Initial', region=a.sets['BOT_RP'], localCsys=None)

	# ############### #
	# ### MESHING ### #
	# ############### #
	a.makeIndependent(instances=(a.instances['Shell-1'], ))
	if order == 1:
		elemType1 = mesh.ElemType(elemCode=S4R5, elemLibrary=STANDARD)
		elemType2 = mesh.ElemType(elemCode=STRI3, elemLibrary=STANDARD)
	if order == 2:
		elemType1 = mesh.ElemType(elemCode=S8R5, elemLibrary=STANDARD)
		elemType2 = mesh.ElemType(elemCode=STRI65, elemLibrary=STANDARD)		
	if meshing == 'TRI': a.setMeshControls(regions=f.findAt(((0.0, 0.5 * L, R), )), elemShape=TRI)
	if meshing == 'QUAD': a.setMeshControls(regions=f.findAt(((0.0, 0.5 * L, R), )), elemShape=QUAD)
	a.setElementType(regions=a.sets['Global_Surface_Set'], elemTypes=(elemType1, elemType2))
	partInstances =(a.instances['Shell-1'], )
	a.seedPartInstance(regions=partInstances, size=2.0 * math.pi * R / float(el_div), deviationFactor=0.1, minSizeFactor=0.1)
	a.generateMesh(regions=partInstances)

	# ################ #
	# ### ANALYSIS ### #
	# ################ #
	if Analysis == "LBA": 
		mdb.models['Model-1'].BuckleStep(name=Analysis, previous='Initial', numEigen=10, eigensolver=LANCZOS, minEigen=0.0, blockSize=DEFAULT, maxBlocks=DEFAULT)
	mdb.models['Model-1'].FieldOutputRequest(name='COORD_OUTPUT', createStepName=Analysis, variables=('U', 'COORD', ))

	counter = 0
	for PROCEDURE in PROCEDURES:
		counter += 1

		# Reset
		for LD in ['UM','UC','UT','TS','BM']:
			try: del mdb.models['Model-1'].loads[LD]
			except: pass

		# Set
		file_root = 'ROT_' + float2str(ROT, 2) + '_O_' + float2str(OMEGA, 2) + '__'
		if PROCEDURE['UM']: # Uniform meridional - downwards point force at top RP
			mdb.models['Model-1'].ConcentratedForce(name='UM', createStepName='LBA', region=a.sets['TOP_RP'], cf2=-LBA_REPOSITORY['UM'], distributionType=UNIFORM, field='', localCsys=None)
			file_root += 'UM_'
		if PROCEDURE['UC']: # Uniform circumferential - uniform inward pressure everywhere
			mdb.models['Model-1'].Pressure(name='UC', createStepName='LBA', region=a.surfaces['Global_Surface'], distributionType=UNIFORM, field='', magnitude=LBA_REPOSITORY['UC'])
			file_root += 'UC_'
		if PROCEDURE['UT']: # Uniform torsion - point moment at top RP
			mdb.models['Model-1'].Moment(name='UT', createStepName='LBA', region=a.sets['TOP_RP'], cm2=LBA_REPOSITORY['UT'], distributionType=UNIFORM, field='', localCsys=None)
			file_root += 'UT_'
		if PROCEDURE['TS']: # Uniform transverse shear - transverse point force at top RP
			mdb.models['Model-1'].ConcentratedForce(name='TS', createStepName='LBA', region=a.sets['TOP_RP'], cf1=LBA_REPOSITORY['TS'], distributionType=UNIFORM, field='', localCsys=None)
			file_root += 'TS_'
		if PROCEDURE['BM']: # Uniform bending moment - point moment at top RP
			mdb.models['Model-1'].Moment(name='BM', createStepName='LBA', region=a.sets['TOP_RP'], cm3=-LBA_REPOSITORY['BM'], distributionType=UNIFORM, field='', localCsys=None)
			file_root += 'BM_'
		file_root = file_root[:-1]
		print("Running: " + file_root)

		# Execution code
		mdb.Job(name=file_root, model='Model-1', description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
			memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
			modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
		mdb.jobs[file_root].writeInput(consistencyChecking=OFF)
		commandS = abq_command + ' job=' + file_root + ' double ask_delete=OFF cpus=' + str(cpus) + ' int' # Submit to /Standard solver
		os.system(commandS) # Submit & analyse

		# Read .odb file and process to obtain real LBA LPFs for undividual fundamental loads
		single = PROCEDURE['UM'] + PROCEDURE['UC'] + PROCEDURE['UT'] + PROCEDURE['TS'] + PROCEDURE['BM']
		try:
			if single == 1: 
				myOdb = openOdb(file_root + '.odb', readOnly=True)
				LPF = float(myOdb.steps['LBA'].frames[1].description.split(' ')[-1])
				if PROCEDURE['UM']: LBA_REPOSITORY['UM'] = LPF
				if PROCEDURE['UC']: LBA_REPOSITORY['UC'] = LPF
				if PROCEDURE['UT']: LBA_REPOSITORY['UT'] = LPF
				if PROCEDURE['TS']: LBA_REPOSITORY['TS'] = LPF
				if PROCEDURE['BM']: LBA_REPOSITORY['BM'] = LPF
				myOdb.close()
		except: break

		# Process deformations from .odb output database to .jpg image
		deformations_to_image(file_root, path_to_Py3, MaxPixelDim, LBA_eigenmodes, Analysis, "Shell".upper()+"-1", "Global_Surface_Set".upper())

		# File housekeeping
		if single == 1:
			for mode in range(1,LBA_eigenmodes + 1):
				if PROCEDURE['UM']: 
					try: shutil.move(file_root + '_' + str(mode) + '.jpg', Analysis + '_Images/UM/' + file_root + '_' + str(mode) + '.jpg')
					except: pass
				if PROCEDURE['UC']: 
					try: shutil.move(file_root + '_' + str(mode) + '.jpg', Analysis + '_Images/UC/' + file_root + '_' + str(mode) + '.jpg')
					except: pass
				if PROCEDURE['UT']: 
					try: shutil.move(file_root + '_' + str(mode) + '.jpg', Analysis + '_Images/UT/' + file_root + '_' + str(mode) + '.jpg')
					except: pass
				if PROCEDURE['TS']: 
					try: shutil.move(file_root + '_' + str(mode) + '.jpg', Analysis + '_Images/TS/' + file_root + '_' + str(mode) + '.jpg')
					except: pass
				if PROCEDURE['BM']: 
					try: shutil.move(file_root + '_' + str(mode) + '.jpg', Analysis + '_Images/BM/' + file_root + '_' + str(mode) + '.jpg')							
					except: pass
		else:
			for mode in range(1,LBA_eigenmodes + 1): 
				shutil.move(file_root + '_' + str(mode) + '.jpg', Analysis + '_Images/' + file_root + '_' + str(mode) + '.jpg')
		if Analysis == 'LBA': exts = ['.inp']#, '.prt', '.fil'] # Files to retain
		for ext in exts:
			try: shutil.move(file_root + ext, Analysis + '_Repository/' + file_root + ext)
			except: pass
		
		if Analysis == 'LBA': exts = ['.bin', '.com', '.dat', '.ipm', '.log', '.odb','.prt', '.msg', '.sim', '.sta'] # Files to remove
		for ext in exts:
			if ext == '.bin':
				for mode in range(1,LBA_eigenmodes + 1): 
					try: os.remove(file_root + '_' + str(mode) + ext)
					except: pass
			else:
				try: os.remove(file_root + ext)
				except: pass