from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import *
from driverUtils import executeOnCaeStartup
from odbAccess import *
import os
from APy2_ABQDeformations_to_Binary import *
executeOnCaeStartup()

### Master generating script for the wind-loaded cylinder example ### 
### Can be run directly from cmd as: abaqus cae noGUI=APy2_Cylinder_Wind.py ###

# (C) Dr Adam Jan Sadowski of Imperial College London, last modified at 19.38 on 31/05/22
# Copyright under a BSD 3-Clause License, see https://github.com/SadowskiAJ/LBANet.git


##############
### INPUTS ###
##############
# Geometry
RADIUS = 100.0 # Lesired radius r, (float) [mm]
THICKNESS = 1.0 # Desired thickness t, (float) [mm]
OMEGAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00] # List of dimensionless length L/r ratios, (float) [-]

# Anchorage
C_SYMM = True # Circumferential symmetry? Bool [-]
Top_BC = 'BC2r' # Top boundary condition, (char) - choose from BC1f, BC1r, BC2f, BC2r
Bot_BC = 'BC1r' # Base boundary condition, (char) - choose from BC1f, BC1r, BC2f, BC2r

# Loading
Ref_Magnitude = 1.0 # Reference magnitude in either case, (float) [N/mm2]

# Material
Elastic_Modulus = 200000.0 # Modulues of elasticity E, (float) [N/mm2 = MPa]
Poisson_Ratio = 0.3 # Poisson ratio nu, (float) [-]
Material_Name = "Generic" # Material name, (char)

# Analysis
LBA_eigenmodes = 1 # For reporting, actual extracted is given as 10 below for analysis stability
path_to_Py3 = """C:\\Users\\"Adam Jan Sadowski"\\AppData\\Local\\Programs\\Python\\Python37\\python.exe"""
abq_command = "abq2017"
MaxPixelDim = 1000

# Meshing
NO_ELS_PER_HALF_CIRC = 60 # No. of elements per half-circumference, (int) [-]
Min_El_Frac = 0.05 # Minimum element size as a fraction of the relevant half-wavelength, (float) [-]
Max_El_Frac = 0.05 # Maximum element size as a fraction of the relevant half-wavelength, (float) [-]
Mesh_Order = 2 # Order of mesh (1 - S4R elements; 2 - S8R elements), (int) [-]


##################################################################
### HEREAFTER DO NOT MODIFY UNLESS YOU KNOW WHAT YOU ARE DOING ###
##################################################################
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

# Main computational loop where the L/r ratio is varied
count = 1
for OMEGA in OMEGAS: 
	print("#### Analysis " + str(count) + " of " + str(len(OMEGAS)) + " ####")
	Mdb()
	session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
	cpus = 1
	Model_Name = "Model-1"
	Sketch_Name = 'Cylinder_Sketch'
	Part_Name = 'Cylinder_Part'
	Section_Name = 'Cylinder_Section'
	HEIGHT = OMEGA * RADIUS * (RADIUS / THICKNESS)**0.5 # Height of shell h, [mm]

	################
	### GEOMETRY ###
	################
	mdb.models[Model_Name].ConstrainedSketch(name=Sketch_Name, sheetSize=RADIUS)
	mdb.models[Model_Name].sketches[Sketch_Name].ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
	mdb.models[Model_Name].sketches[Sketch_Name].Line(point1=(RADIUS, 0.0), point2=(RADIUS, HEIGHT))
	mdb.models[Model_Name].Part(dimensionality=THREE_D, name=Part_Name, type=DEFORMABLE_BODY)
	if C_SYMM: mdb.models[Model_Name].parts[Part_Name].BaseShellRevolve(sketch=mdb.models[Model_Name].sketches[Sketch_Name], angle=180.0, flipRevolveDirection=OFF)
	else: mdb.models[Model_Name].parts[Part_Name].BaseShellRevolve(sketch=mdb.models[Model_Name].sketches[Sketch_Name], angle=360.0, flipRevolveDirection=OFF)

	################
	### MATERIAL ###
	################
	mdb.models[Model_Name].Material(description=Material_Name, name=Material_Name)
	mdb.models[Model_Name].materials[Material_Name].Elastic(table=((Elastic_Modulus, Poisson_Ratio), ))
	mdb.models[Model_Name].HomogeneousShellSection(idealization=NO_IDEALIZATION, integrationRule=SIMPSON, material=Material_Name,
		name=Section_Name, nodalThicknessField='', numIntPts=5, poissonDefinition=DEFAULT, preIntegrate=OFF, temperature=GRADIENT,
		thickness=THICKNESS, thicknessField='', thicknessModulus=None, thicknessType=UNIFORM, useDensity=OFF)
	p = mdb.models[Model_Name].parts[Part_Name]
	faces = p.faces.findAt(((RADIUS, 0.5 * HEIGHT, 0.0), ))
	region = p.Set(faces=faces, name="Global_Surface_Set")
	p.SectionAssignment(region=region, sectionName=Section_Name, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

	##################################################################
	### ASSEMBLY, PARTITIONING, CONSTRAINTS AND SET IDENTIFICATION ###
	##################################################################
	mdb.models[Model_Name].rootAssembly.DatumCsysByDefault(CARTESIAN)
	mdb.models[Model_Name].rootAssembly.Instance(dependent=OFF, name=Part_Name+"-1", part=mdb.models[Model_Name].parts[Part_Name])
	a = mdb.models[Model_Name].rootAssembly
	Cyl_CSYS = a.DatumCsysByThreePoints(name='Cylindrical_CSYS', coordSysType=CYLINDRICAL, origin=(0.0, 0.0, 0.0), line1=(1.0, 0.0, 0.0), line2=(0.0, 0.0, -1.0))
	Cyl_CSYS = mdb.models['Model-1'].rootAssembly.datums[Cyl_CSYS.id]
	a.features['Datum csys-1'].suppress()

	# Global surface
	a.Surface(side2Faces=a.instances[Part_Name+"-1"].faces.findAt(((RADIUS, 0.5 * HEIGHT, 0.0), )), name='Global_Surface')
	a.Set(faces=a.instances[Part_Name+"-1"].faces.findAt(((RADIUS, 0.5 * HEIGHT, 0.0), )), name='Global_Surface_Set')

	# Free Edge Sets
	Cedge_BOT = a.instances[Part_Name+"-1"].edges.findAt(((0.0, 0.0, RADIUS), ))
	Cedge_TOP = a.instances[Part_Name+"-1"].edges.findAt(((0.0, HEIGHT, RADIUS), ))	
	a.Set(edges=Cedge_BOT, name='Circ_Edge_Bot')
	a.Set(edges=Cedge_TOP, name='Circ_Edge_Top')
	if C_SYMM:
		Medge_LHS = a.instances[Part_Name+"-1"].edges.findAt(((-RADIUS, 0.5 * HEIGHT, 0.0), ))
		Medge_RHS = a.instances[Part_Name+"-1"].edges.findAt(((RADIUS, 0.5 * HEIGHT, 0.0), ))
		a.Set(edges=Medge_LHS, name='Mer_Edge_LHS')
		a.Set(edges=Medge_RHS, name='Mer_Edge_RHS')	

	#########################
	### LBA ANALYSIS STEP ###
	#########################
	StepName = "LBA"
	mdb.models[Model_Name].BuckleStep(blockSize=DEFAULT, eigensolver=LANCZOS, maxBlocks=DEFAULT, minEigen=0.0, name=StepName, numEigen=10, previous='Initial')
	mdb.models[Model_Name].FieldOutputRequest(name='COORD_OUTPUT', createStepName=StepName, variables=('U', 'COORD', ))

	###################
	### LOADS & BCS ###
	###################
	BCs = {'BC1r' : [SET,SET,SET,SET,SET,SET],
		'BC1f' : [SET,SET,SET,SET,UNSET,SET],
		'BC2r' : [SET,SET,UNSET,SET,SET,SET],
		'BC2f' : [SET,SET,UNSET,SET,UNSET,SET],
		'C_SYMM' : [UNSET,SET,UNSET,SET,UNSET,SET]}

	mdb.models[Model_Name].DisplacementBC(name='Bot_'+Bot_BC, createStepName=StepName, region=a.sets['Circ_Edge_Bot'], u1=BCs[Bot_BC][0], u2=BCs[Bot_BC][1], u3=BCs[Bot_BC][2], ur1=BCs[Bot_BC][3], ur2=BCs[Bot_BC][4], ur3=BCs[Bot_BC][5], amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=Cyl_CSYS)
	mdb.models[Model_Name].DisplacementBC(name='Top_'+Top_BC, createStepName=StepName, region=a.sets['Circ_Edge_Top'], u1=BCs[Top_BC][0], u2=BCs[Top_BC][1], u3=BCs[Top_BC][2], ur1=BCs[Top_BC][3], ur2=BCs[Top_BC][4], ur3=BCs[Top_BC][5], amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=Cyl_CSYS)
	if C_SYMM:
		mdb.models[Model_Name].DisplacementBC(name='CSymm_LHS', createStepName=StepName, region=a.sets['Mer_Edge_LHS'], u1=BCs['C_SYMM'][0], u2=BCs['C_SYMM'][1], u3=BCs['C_SYMM'][2], ur1=BCs['C_SYMM'][3], ur2=BCs['C_SYMM'][4], ur3=BCs['C_SYMM'][5], amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=Cyl_CSYS)
		mdb.models[Model_Name].DisplacementBC(name='CSymm_RHS', createStepName=StepName, region=a.sets['Mer_Edge_RHS'], u1=BCs['C_SYMM'][0], u2=BCs['C_SYMM'][1], u3=BCs['C_SYMM'][2], ur1=BCs['C_SYMM'][3], ur2=BCs['C_SYMM'][4], ur3=BCs['C_SYMM'][5], amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=Cyl_CSYS)

	# Circumferentially-varying wind pressure as per EN 1993-4-1 Annex C
	dcoL = "("+str(2.0 * RADIUS / HEIGHT)+")"
	mdb.models[Model_Name].ExpressionField(name='Wind_Pressure_Field', localCsys=Cyl_CSYS, description='', expression='-0.54 + 0.16 * ' + dcoL + ' + (0.28 + 0.04 * ' + dcoL + ') * cos(Th) + (1.04 - 0.20 * ' + dcoL + ') * cos(2 * Th) + (0.36 - 0.05 * ' + dcoL + ') * cos(3 * Th) - (0.14 - 0.05 * ' + dcoL + ') * cos(4 * Th)')
	mdb.models[Model_Name].Pressure(name='Wind_Pressure', createStepName=StepName, region=a.surfaces['Global_Surface'], distributionType=FIELD, field='Wind_Pressure_Field', magnitude=-Ref_Magnitude)

	###############
	### MESHING ###
	###############
	if C_SYMM: a.setMeshControls(regions=a.instances[Part_Name+"-1"].sets['Global_Surface_Set'].faces, elemShape=QUAD, technique=STRUCTURED)
	else: a.setMeshControls(regions=a.instances[Part_Name+"-1"].sets['Global_Surface_Set'].faces, elemShape=QUAD, technique=SWEEP)		
	a.seedEdgeBySize(edges=a.sets['Circ_Edge_Bot'].edges, size=pi * RADIUS / float(NO_ELS_PER_HALF_CIRC), deviationFactor=0.1, constraint=FINER)
	a.seedEdgeBySize(edges=a.sets['Circ_Edge_Top'].edges, size=pi * RADIUS / float(NO_ELS_PER_HALF_CIRC), deviationFactor=0.1, constraint=FINER)	
	a.makeIndependent(instances=(a.instances[Part_Name+"-1"], ))
	if Mesh_Order == 1:
		elemType1 = mesh.ElemType(elemCode=S4R5, elemLibrary=STANDARD)
		elemType2 = mesh.ElemType(elemCode=STRI3, elemLibrary=STANDARD)
	if Mesh_Order == 2:
		elemType1 = mesh.ElemType(elemCode=S8R5, elemLibrary=STANDARD)
		elemType2 = mesh.ElemType(elemCode=STRI65, elemLibrary=STANDARD)		
	a.setMeshControls(regions=a.instances[Part_Name+"-1"].faces.findAt(((0.0, 0.5 * HEIGHT, RADIUS), )), elemShape=QUAD)
	a.setElementType(regions=a.sets['Global_Surface_Set'], elemTypes=(elemType1, elemType2))
	a.generateMesh(regions=(a.instances[Part_Name+"-1"], ))
	
	################
	### ANALYSIS ###
	################
	# Write .inp file
	file_root = StepName + "_R" + float2str(RADIUS, 2) + "_t" + float2str(THICKNESS, 2) + "_O" + float2str(OMEGA, 2) + "_Wind"
	mdb.Job(name=file_root, model=Model_Name, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
		memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=DOUBLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
		modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=2, numDomains=2, numGPUs=0) 
	mdb.jobs[file_root].writeInput(consistencyChecking=OFF)

	# Submit to /Standard sovler for analysis
	commandS = abq_command + ' job=' + file_root + ' double ask_delete=OFF cpus=' + str(cpus) + ' int' 
	os.system(commandS) 
	print("LBA complete.")
	count += 1

	# Process deformations from .odb output database to .jpg chromatic signature image via intermediate .bin file
	deformations_to_image(file_root, path_to_Py3, MaxPixelDim, LBA_eigenmodes, StepName, Part_Name.upper()+"-1", "Global_Surface_Set".upper())

	# Read in eigenvalue
	myOdb = openOdb(file_root + '.odb', readOnly=True)
	for E in range(1,LBA_eigenmodes+1):
		fid = open(file_root + "_" + str(E) + ".txt",'w')
		LPF = float(myOdb.steps['LBA'].frames[E].description.split(' ')[-1])
		fid.write(str(LPF) + "\n")
	fid.close()
	myOdb.close()

	# Cleanup
	exts = ['.bin', '.com', '.dat', '.inp', '.ipm', '.log', '.prt', '.msg', '.sim', '.sta'] # Files to remove
	for ext in exts:
		if ext == '.bin':
			for mode in range(1,LBA_eigenmodes + 1): 
				try: os.remove(file_root + '_' + str(mode) + ext)
				except: pass
		else:
			try: os.remove(file_root + ext)
			except: pass	

	##############################
	### DYNAMIC CLASSIFICATION ###
	##############################
	# Call Python 3 processing script via system call (while clearing python path) to classify chromatic signature
	tmp = os.environ["PYTHONPATH"]
	os.environ["PYTHONPATH"] = ""
	commandS = path_to_Py3 + " Py3_Classify_Signature.py " + file_root + " " + str(LBA_eigenmodes)
	os.system(commandS)
	os.environ["PYTHONPATH"] = tmp

	# Read class from .txt file and display to console
	for E in range(1,LBA_eigenmodes+1):
		fid = open(file_root + "_" + str(E) + ".txt",'r')
		LBA_LPF = fid.readline()[0:-1]
		classified = fid.readline()
		print("Eigenmode " + str(E) + " with eigenvalue " + str(LBA_LPF) + " has a chromatic signature classified as: " + classified + "\n\n")
		fid.close()