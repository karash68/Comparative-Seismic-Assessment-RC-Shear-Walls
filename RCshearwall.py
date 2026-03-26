
# =============================================================================
# RC SHEAR WALL FINITE ELEMENT MODEL - OpenSeesPy
# =============================================================================
# Converted to openseespy by: Anurag Upadhyay, University of Utah.
# Units: N and m to follow the originally published code.
#
# MODEL DESCRIPTION:
# - 2D RC shear wall (1m wide × 2m tall × 0.125m thick)
# - Shell elements with layered sections (concrete + rebar)
# - Boundary elements (confined) + Interior elements (unconfined)
# - P-Delta effects included via truss columns
# - Nonlinear materials: PSUMAT concrete + Steel02 rebar
# - Analysis types: Gravity, Pushover, Cyclic loading
# =============================================================================

from openseespy.opensees import *

import numpy as np
import matplotlib.pyplot as plt
import os
import math

pi = 3.1415

# ANALYSIS TYPE SELECTION
# Options: "Gravity", "Pushover", "Cyclic"
AnalysisType = "Pushover" # Cyclic   Pushover   Gravity

wipe()

model('basic','-ndm',3,'-ndf',6)

###################################
## MATERIAL DEFINITIONS
###################################

# =============================================================================
# CONCRETE MATERIAL - PSUMAT (Penn State University Material Model)
# =============================================================================
# Material ID 1: Advanced concrete model with cracking, crushing, and cyclic behavior
# Parameters: ModelID=40 (PSUMAT), NumParams=7, fc'=20.7MPa, ft=2.07MPa, 
#            fcu=-4.14MPa, εc0=-0.002, εcu=-0.01, εt=0.001, ν=0.3
nDMaterial('PlaneStressUserMaterial',1,40,7,20.7e6,2.07e6,-4.14e6,-0.002,-0.01,0.001,0.3)

# Material ID 4: Convert concrete to plate/shell format with out-of-plane modulus
# Parameters: PlaneStressMaterial=1, OutOfPlaneModulus=1.25e10 Pa
nDMaterial('PlateFromPlaneStress',4,1,1.25e10)

# =============================================================================
# STEEL REBAR MATERIALS - Steel02 (Giuffre-Menegotto-Pinto model)
# =============================================================================
# Material ID 7: Steel for boundary element rebar (fy=379MPa, E=202.7GPa)
# Parameters: fy=379MPa, E0=202.7GPa, b=0.01 (strain hardening), 
#            Bauschinger parameters: a1=18.5, a2=0.925, a4=0.15
uniaxialMaterial('Steel02',7,379e6,202.7e9,0.01,18.5,0.925,0.15)

# Material ID 8: Steel for interior element rebar (fy=392MPa, E=200.6GPa)
# Slightly different properties for interior reinforcement
uniaxialMaterial('Steel02',8,392e6,200.6e9,0.01,18.5,0.925,0.15)

# =============================================================================
# PLATE REBAR MATERIALS - Convert 1D steel to 2D plate rebar
# =============================================================================
# Material ID 9: Horizontal rebar (90° angle) - boundary elements
nDMaterial('PlateRebar',9,7,90.0)  # Steel02 material 7, 90° orientation

# Material ID 10: Horizontal rebar (90° angle) - interior elements  
nDMaterial('PlateRebar',10,8,90.0) # Steel02 material 8, 90° orientation

# Material ID 11: Vertical rebar (0° angle) - both boundary and interior
nDMaterial('PlateRebar',11,8,0.0)  # Steel02 material 8, 0° orientation

# =============================================================================
# LAYERED SHELL SECTIONS - Multi-layer RC wall cross-sections
# =============================================================================

# SECTION 1: BOUNDARY ELEMENTS (Confined concrete zones at wall edges)
# 10 layers total, thickness = ~125mm
# Layer stack (bottom to top):
#   Layer 1:  Concrete cover         (Mat 4,  t=12.5mm)
#   Layer 2:  Vertical rebar         (Mat 11, t=0.24mm)
#   Layer 3:  Vertical rebar         (Mat 11, t=0.37mm)
#   Layer 4:  Concrete core          (Mat 4,  t=24.7mm)
#   Layer 5:  Concrete core          (Mat 4,  t=24.7mm)
#   Layer 6:  Concrete core          (Mat 4,  t=24.7mm)
#   Layer 7:  Concrete core          (Mat 4,  t=24.7mm)
#   Layer 8:  Vertical rebar         (Mat 11, t=0.37mm)
#   Layer 9:  Vertical rebar         (Mat 11, t=0.24mm)
#   Layer 10: Concrete cover         (Mat 4,  t=12.5mm)
section('LayeredShell',1,10,4,0.0125,11,0.0002403,11,0.0003676,4,0.024696,4,0.024696,4,0.024696,4,0.024696,11,0.0003676,11,0.0002403,4,0.0125)

# SECTION 2: INTERIOR ELEMENTS (Unconfined concrete zones in wall center)
# 8 layers total, thickness = ~125mm
# Layer stack (bottom to top):
#   Layer 1: Concrete cover          (Mat 4,  t=12.5mm)
#   Layer 2: Vertical rebar          (Mat 11, t=0.24mm)
#   Layer 3: Horizontal rebar        (Mat 10, t=0.24mm)
#   Layer 4: Concrete core           (Mat 4,  t=49.5mm)
#   Layer 5: Concrete core           (Mat 4,  t=49.5mm)
#   Layer 6: Horizontal rebar        (Mat 10, t=0.24mm)
#   Layer 7: Vertical rebar          (Mat 11, t=0.24mm)
#   Layer 8: Concrete cover          (Mat 4,  t=12.5mm)
section('LayeredShell',2,8,4,0.0125,11,0.0002403,10,0.0002356,4,0.0495241,4,0.0495241,10,0.0002356,11,0.0002403,4,0.0125)

# =============================================================================
# NODE DEFINITIONS - 5×11 rectangular grid
# =============================================================================
# Wall dimensions: 1.0m wide × 2.0m tall
# Grid: 5 nodes horizontally × 11 nodes vertically = 55 total nodes
# X-coordinates: 0.0, 0.2, 0.5, 0.8, 1.0m (non-uniform spacing)
# Y-coordinates: 0.0 to 2.0m in 0.2m increments
# Z-coordinates: 0.0 (all nodes in X-Y plane)
#
# Node numbering pattern:
# 51  52  53  54  55  ← Row 11 (y=2.0m, top)
# 46  47  48  49  50  ← Row 10 (y=1.8m)
# ...  ...  ...  ... 
# 6   7   8   9   10  ← Row 2  (y=0.2m)
# 1   2   3   4   5   ← Row 1  (y=0.0m, bottom - fixed)
# =============================================================================

# ROW 1: Bottom nodes (y = 0.0m) - FIXED BASE
node(1,0.0,0,0)  # Bottom-left corner
node(2,0.2,0,0)  # 
node(3,0.5,0,0)  # Center-bottom
node(4,0.8,0,0)  # 
node(5,1.0,0,0)  # Bottom-right corner

# ROW 2: y = 0.2m
node(6,0.0,0.2,0)
node(7,0.2,0.2,0)
node(8,0.5,0.2,0)
node(9,0.8,0.2,0)
node(10,1.0,0.2,0)

# ROW 3: y = 0.4m
node(11,0.0,0.4,0)
node(12,0.2,0.4,0)
node(13,0.5,0.4,0)
node(14,0.8,0.4,0)
node(15,1.0,0.4,0)

# ROW 4: y = 0.6m
node(16,0.0,0.6,0)
node(17,0.2,0.6,0)
node(18,0.5,0.6,0)
node(19,0.8,0.6,0)
node(20,1.0,0.6,0)

# ROW 5: y = 0.8m
node(21,0.0,0.8,0)
node(22,0.2,0.8,0)
node(23,0.5,0.8,0)
node(24,0.8,0.8,0)
node(25,1.0,0.8,0)
                 
# ROW 6: y = 1.0m (mid-height)
node(26,0.0,1.0,0)
node(27,0.2,1.0,0)
node(28,0.5,1.0,0)
node(29,0.8,1.0,0)
node(30,1.0,1.0,0)

# ROW 7: y = 1.2m
node(31,0.0,1.2,0)
node(32,0.2,1.2,0)
node(33,0.5,1.2,0)
node(34,0.8,1.2,0)
node(35,1.0,1.2,0)
                 
# ROW 8: y = 1.4m
node(36,0.0,1.4,0)
node(37,0.2,1.4,0)
node(38,0.5,1.4,0)
node(39,0.8,1.4,0)
node(40,1.0,1.4,0)

# ROW 9: y = 1.6m
node(41,0.0,1.6,0)
node(42,0.2,1.6,0)
node(43,0.5,1.6,0)
node(44,0.8,1.6,0)
node(45,1.0,1.6,0)
                 
# ROW 10: y = 1.8m
node(46,0.0,1.8,0)
node(47,0.2,1.8,0)
node(48,0.5,1.8,0)
node(49,0.8,1.8,0)
node(50,1.0,1.8,0)
                
# ROW 11: Top nodes (y = 2.0m) - LOAD APPLICATION POINT
node(51,0.0,2.0,0)  # Top-left corner
node(52,0.2,2.0,0)  #
node(53,0.5,2.0,0)  # Top-center (CONTROL NODE for pushover)
node(54,0.8,2.0,0)  #
node(55,1.0,2.0,0)  # Top-right corner

# =============================================================================
# SHELL ELEMENT DEFINITIONS - 4×10 mesh = 40 total elements
# =============================================================================
# Element type: ShellNLDKGQ (Non-Linear Discrete Kirchhoff Quadrilateral)
# Alternative: ShellMITC4 (Mixed Interpolation of Tensorial Components)
# 
# Section assignment pattern:
# - Boundary columns (x=0.0-0.2m and x=0.8-1.0m): Section 1 (confined)
# - Interior columns (x=0.2-0.8m): Section 2 (unconfined)
#
# Element numbering: Row-by-row from bottom to top
# Each element connects 4 nodes in counter-clockwise order
# =============================================================================

# SHELL ELEMENT TYPE SELECTION
ShellType = "ShellNLDKGQ"  # Advanced nonlinear formulation for RC analysis
# ShellType = "ShellMITC4"  # Alternative: simpler formulation

# ROW 1 ELEMENTS (y = 0.0 to 0.2m)
# Element 1: Left boundary (Section 1 - confined)
element(ShellType,1,1,2,7,6,1)    # Nodes: 1→2→7→6, Section 1
# Element 2: Left-center interior (Section 2 - unconfined)
element(ShellType,2,2,3,8,7,2)    # Nodes: 2→3→8→7, Section 2
# Element 3: Right-center interior (Section 2 - unconfined)
element(ShellType,3,3,4,9,8,2)    # Nodes: 3→4→9→8, Section 2
# Element 4: Right boundary (Section 1 - confined)
element(ShellType,4,4,5,10,9,1)   # Nodes: 4→5→10→9, Section 1

# ROW 2 ELEMENTS (y = 0.2 to 0.4m)
element(ShellType,5,6,7,12,11,1)     # Left boundary
element(ShellType,6,7,8,13,12,2)     # Left-center interior
element(ShellType,7,8,9,14,13,2)     # Right-center interior
element(ShellType,8,9,10,15,14,1)    # Right boundary

# ROW 3 ELEMENTS (y = 0.4 to 0.6m)
element(ShellType,9,11,12,17,16,1)   # Left boundary
element(ShellType,10,12,13,18,17,2)  # Left-center interior
element(ShellType,11,13,14,19,18,2)  # Right-center interior
element(ShellType,12,14,15,20,19,1)  # Right boundary

# ROW 4 ELEMENTS (y = 0.6 to 0.8m)
element(ShellType,13,16,17,22,21,1)  # Left boundary
element(ShellType,14,17,18,23,22,2)  # Left-center interior
element(ShellType,15,18,19,24,23,2)  # Right-center interior
element(ShellType,16,19,20,25,24,1)  # Right boundary

# ROW 5 ELEMENTS (y = 0.8 to 1.0m)
element(ShellType,17,21,22,27,26,1)  # Left boundary
element(ShellType,18,22,23,28,27,2)  # Left-center interior
element(ShellType,19,23,24,29,28,2)  # Right-center interior
element(ShellType,20,24,25,30,29,1)  # Right boundary

# ROW 6 ELEMENTS (y = 1.0 to 1.2m) - Mid-height
element(ShellType,21,26,27,32,31,1)  # Left boundary
element(ShellType,22,27,28,33,32,2)  # Left-center interior
element(ShellType,23,28,29,34,33,2)  # Right-center interior
element(ShellType,24,29,30,35,34,1)  # Right boundary

# ROW 7 ELEMENTS (y = 1.2 to 1.4m)
element(ShellType,25,31,32,37,36,1)  # Left boundary
element(ShellType,26,32,33,38,37,2)  # Left-center interior
element(ShellType,27,33,34,39,38,2)  # Right-center interior
element(ShellType,28,34,35,40,39,1)  # Right boundary

# ROW 8 ELEMENTS (y = 1.4 to 1.6m)
element(ShellType,29,36,37,42,41,1)  # Left boundary
element(ShellType,30,37,38,43,42,2)  # Left-center interior
element(ShellType,31,38,39,44,43,2)  # Right-center interior
element(ShellType,32,39,40,45,44,1)  # Right boundary

# ROW 9 ELEMENTS (y = 1.6 to 1.8m)
element(ShellType,33,41,42,47,46,1)  # Left boundary
element(ShellType,34,42,43,48,47,2)  # Left-center interior
element(ShellType,35,43,44,49,48,2)  # Right-center interior
element(ShellType,36,44,45,50,49,1)  # Right boundary

# ROW 10 ELEMENTS (y = 1.8 to 2.0m) - Top row
element(ShellType,37,46,47,52,51,1)  # Left boundary
element(ShellType,38,47,48,53,52,2)  # Left-center interior
element(ShellType,39,48,49,54,53,2)  # Right-center interior
element(ShellType,40,49,50,55,54,1)  # Right boundary

# =============================================================================
# P-DELTA TRUSS ELEMENTS - Second-order effects (P-Δ)
# =============================================================================
# Purpose: Capture additional moments due to vertical loads acting on 
#          laterally displaced structure (P-Δ effects)
# Configuration: 4 vertical truss columns at x = 0.0, 0.2, 0.8, 1.0m
# Properties: Very small area (223.53e-6 m² ≈ 0.22 mm²) to avoid 
#            affecting lateral stiffness while carrying axial loads
# Material: Steel02 material ID 7
# =============================================================================

# COLUMN 1: Left edge (x = 0.0m)
element('truss',41,1,6,223.53e-6,7)    # Level 1→2: nodes 1→6
element('truss',42,6,11,223.53e-6,7)   # Level 2→3: nodes 6→11
element('truss',43,11,16,223.53e-6,7)  # Level 3→4: nodes 11→16
element('truss',44,16,21,223.53e-6,7)  # Level 4→5: nodes 16→21
element('truss',45,21,26,223.53e-6,7)  # Level 5→6: nodes 21→26
element('truss',46,26,31,223.53e-6,7)  # Level 6→7: nodes 26→31
element('truss',47,31,36,223.53e-6,7)  # Level 7→8: nodes 31→36
element('truss',48,36,41,223.53e-6,7)  # Level 8→9: nodes 36→41
element('truss',49,41,46,223.53e-6,7)  # Level 9→10: nodes 41→46
element('truss',50,46,51,223.53e-6,7)  # Level 10→11: nodes 46→51

# COLUMN 2: Left-center (x = 0.2m)
element('truss',51,2,7,223.53e-6,7)    # Level 1→2: nodes 2→7
element('truss',52,7,12,223.53e-6,7)   # Level 2→3: nodes 7→12
element('truss',53,12,17,223.53e-6,7)  # Level 3→4: nodes 12→17
element('truss',54,17,22,223.53e-6,7)  # Level 4→5: nodes 17→22
element('truss',55,22,27,223.53e-6,7)  # Level 5→6: nodes 22→27
element('truss',56,27,32,223.53e-6,7)  # Level 6→7: nodes 27→32
element('truss',57,32,37,223.53e-6,7)  # Level 7→8: nodes 32→37
element('truss',58,37,42,223.53e-6,7)  # Level 8→9: nodes 37→42
element('truss',59,42,47,223.53e-6,7)  # Level 9→10: nodes 42→47
element('truss',60,47,52,223.53e-6,7)  # Level 10→11: nodes 47→52

# COLUMN 3: Right-center (x = 0.8m)
element('truss',61,4,9,223.53e-6,7)    # Level 1→2: nodes 4→9
element('truss',62,9,14,223.53e-6,7)   # Level 2→3: nodes 9→14
element('truss',63,14,19,223.53e-6,7)  # Level 3→4: nodes 14→19
element('truss',64,19,24,223.53e-6,7)  # Level 4→5: nodes 19→24
element('truss',65,24,29,223.53e-6,7)  # Level 5→6: nodes 24→29
element('truss',66,29,34,223.53e-6,7)  # Level 6→7: nodes 29→34
element('truss',67,34,39,223.53e-6,7)  # Level 7→8: nodes 34→39
element('truss',68,39,44,223.53e-6,7)  # Level 8→9: nodes 39→44
element('truss',69,44,49,223.53e-6,7)  # Level 9→10: nodes 44→49
element('truss',70,49,54,223.53e-6,7)  # Level 10→11: nodes 49→54

# COLUMN 4: Right edge (x = 1.0m)
element('truss',71,5,10,223.53e-6,7)   # Level 1→2: nodes 5→10
element('truss',72,10,15,223.53e-6,7)  # Level 2→3: nodes 10→15
element('truss',73,15,20,223.53e-6,7)  # Level 3→4: nodes 15→20
element('truss',74,20,25,223.53e-6,7)  # Level 4→5: nodes 20→25
element('truss',75,25,30,223.53e-6,7)  # Level 5→6: nodes 25→30
element('truss',76,30,35,223.53e-6,7)  # Level 6→7: nodes 30→35
element('truss',77,35,40,223.53e-6,7)  # Level 7→8: nodes 35→40
element('truss',78,40,45,223.53e-6,7)  # Level 8→9: nodes 40→45
element('truss',79,45,50,223.53e-6,7)  # Level 9→10: nodes 45→50
element('truss',80,50,55,223.53e-6,7)  # Level 10→11: nodes 50→55

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================
# Fix all bottom nodes (y = 0.0) - simulate foundation/base connection
# Constraints: All 6 DOFs fixed (ux, uy, uz, θx, θy, θz)
fixY(0.0,1,1,1,1,1,1)  # Fix nodes 1,2,3,4,5

# =============================================================================
# RECORDERS - Data output for analysis results
# =============================================================================
# Record horizontal reactions at base nodes for force-displacement curve
recorder('Node','-file','ReactionPY.txt','-time','-node',1,2,3,4,5,'-dof',1,'reaction')

# =============================================================================
# GRAVITY ANALYSIS - Apply vertical loads to simulate dead loads
# =============================================================================
# Purpose: Establish initial stress state before lateral loading
# Load: 320 kN downward at top-center node (53)
# Analysis: Linear static with load control

print("running gravity")

# Define linear time series for gravity loading
timeSeries("Linear", 1)					
# Create load pattern (Pattern ID 1, TimeSeries ID 1)
pattern('Plain',1,1)
# Apply vertical load: 320 kN downward at node 53 (top center)
# Load vector: [Fx=0, Fy=-320kN, Fz=0, Mx=0, My=0, Mz=0]
load(53,0,-320000.0,0.0,0.0,0.0,0.0)

# Record displacement at control node for monitoring
recorder('Node','-file','Disp.txt','-time','-node',53,'-dof',1,'disp')

# ANALYSIS PARAMETERS for gravity analysis
constraints('Plain')                    # Constraint handler: plain (basic)
numberer('RCM')                        # DOF numberer: Reverse Cuthill-McKee
system('BandGeneral')                  # System of equations: band general solver
test('NormDispIncr',1.0e-4,200)       # Convergence test: displacement norm < 1e-4
algorithm('BFGS','-count',100)         # Solution algorithm: BFGS quasi-Newton
integrator('LoadControl',0.1)          # Load control: 0.1 load increment per step
analysis('Static')                     # Analysis type: static
analyze(10)                           # Perform 10 steps (10 × 0.1 = 1.0 full load)

print("gravity analysis complete...")

# Keep gravity loads constant for subsequent analyses
loadConst('-time',0.0)
wipeAnalysis()                        # Clear analysis objects

# =============================================================================
# CYCLIC ANALYSIS - Displacement-controlled cyclic loading
# =============================================================================
if(AnalysisType=="Cyclic"):
	
	# Load-controlled analysis using displacement history from input file
	# File "RCshearwall_Load_input.txt" contains cyclic displacement values
	
	print("<<<< Running Cyclic Analysis >>>>")
	
	# Define path time series from input file
	# dt=0.1: time step, each displacement value applied for 0.1 time units
	timeSeries('Path',2,'-dt',0.1,'-filePath','RCshearwall_Load_input.txt')
	pattern('Plain',2,2)                   # Load pattern using time series 2
	# Single-point constraint: impose displacement at node 53, DOF 1
	sp(53,1,1)                            # Node 53, DOF 1, constraint factor 1

	# ANALYSIS PARAMETERS for cyclic loading
	constraints('Penalty',1e20,1e20)       # Penalty method with large penalty factors
	numberer('RCM')                        # DOF numberer
	system('BandGeneral')                  # Linear system solver
	test('NormDispIncr',1e-05, 100, 1)    # Convergence test: tighter tolerance
	algorithm('KrylovNewton')              # Newton-Krylov algorithm for nonlinear
	integrator('LoadControl',0.1)          # Load control integration
	analysis('Static')                     # Static analysis
	analyze(700)                          # 700 steps (700 displacement points)


# =============================================================================
# PUSHOVER ANALYSIS - Monotonic displacement-controlled loading
# =============================================================================
if(AnalysisType=="Pushover"):
	
	print("<<<< Running Pushover Analysis >>>>")

	# Create load pattern for pushover (unit reference load)
	pattern("Plain", 2, 1)
	
	# PUSHOVER PARAMETERS
	ControlNode=53                        # Control node: top center (node 53)
	ControlDOF=1                         # Control DOF: horizontal (DOF 1)
	MaxDisp= 0.020                       # Maximum displacement: 20mm
	DispIncr=0.00001                     # Displacement increment: 0.01mm per step
	NstepsPush=int(MaxDisp/DispIncr)     # Number of steps: 2000 steps
	
	# Apply unit reference load at control node (will be scaled by load factor)
	load(ControlNode, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0)
	
	# ANALYSIS PARAMETERS for pushover
	system("BandGeneral")                                    # Linear solver
	numberer("RCM")                                         # DOF numberer
	constraints('Penalty',1e20,1e20)                        # Constraint handler
	# Displacement control: control node, DOF, displacement increment
	integrator("DisplacementControl", ControlNode, ControlDOF, DispIncr)
	algorithm('KrylovNewton')                               # Nonlinear algorithm
	test('NormDispIncr',1e-05, 1000, 2)                    # Convergence test
	analysis("Static")                                      # Analysis type
	
	# Create output directory for results
	PushDataDir = r'PushoverResults'
	if not os.path.exists(PushDataDir):
		os.makedirs(PushDataDir)
	
	# RECORDERS for pushover results
	# Base reactions (sum = base shear)
	recorder('Node', '-file', "PushoverOut/React.out", '-closeOnWrite', '-node', 1, 2, 3, 4, 5, '-dof',1, 'reaction')
	# Top displacement (control node)
	recorder('Node', '-file', "PushoverOut/Disp.out", '-closeOnWrite', '-node', ControlNode, '-dof',1, 'disp')

	# PERFORM PUSHOVER ANALYSIS
	# Store results: [displacement(mm), base_shear(kN)]
	dataPush = np.zeros((NstepsPush+1,5))
	for j in range(NstepsPush):
		analyze(1)                                          # Analyze one step
		dataPush[j+1,0] = nodeDisp(ControlNode,1)*1000     # Displacement [mm]
		dataPush[j+1,1] = -getLoadFactor(2)*0.001          # Base shear [kN]
		
	# =============================================================================
	# POST-PROCESSING: Create pushover curve and compare with test data
	# =============================================================================
	# Load experimental test data for validation
	Test = np.loadtxt("RCshearwall_TestOutput.txt", delimiter="\t", unpack="False")
	
	# PLOT SETTINGS
	plt.rcParams.update({'font.size': 7})
	plt.figure(figsize=(4,3), dpi=100)
	plt.rc('font', family='serif')
	
	# PLOT COMPARISON: Test vs Numerical
	plt.plot(Test[0,:], Test[1,:], color="black", linewidth=0.8, linestyle="--", label='Test')
	plt.plot(dataPush[:,0], -dataPush[:,1], color="red", linewidth=1.2, linestyle="-", label='Pushover')
	
	# PLOT FORMATTING
	plt.axhline(0, color='black', linewidth=0.4)          # Zero line (horizontal)
	plt.axvline(0, color='black', linewidth=0.4)          # Zero line (vertical)
	plt.xlim(-25, 25)                                     # X-axis limits
	plt.xticks(np.linspace(-20,20,11,endpoint=True))      # X-axis ticks
	plt.grid(linestyle='dotted')                          # Grid
	plt.xlabel('Displacement (mm)')                       # X-axis label
	plt.ylabel('Base Shear (kN)')                         # Y-axis label
	plt.legend()                                          # Legend
	
	# SAVE AND DISPLAY
	plt.savefig("PushoverOut/RCshearwall_PushoverCurve.png",dpi=1200)
	plt.show()
	
	# Display pushover results in terminal
	print("\nPushover Results (Displacement [mm], Base Shear [kN]):")
	# Print first 5 and last 5 results for summary
	n_show = 5
	for i in range(n_show):
		print(f"Step {i}: Disp = {dataPush[i,0]:.4f} mm, Shear = {dataPush[i,1]:.2f} kN")
	print("...")
	for i in range(-n_show,0):
		idx = dataPush.shape[0] + i
		print(f"Step {idx}: Disp = {dataPush[idx,0]:.4f} mm, Shear = {dataPush[idx,1]:.2f} kN")
	print("Pushover analysis complete")

# =============================================================================
# END OF MODEL
# =============================================================================
