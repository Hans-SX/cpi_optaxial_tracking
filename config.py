# Selecting folders
STD_PATH   = False             # True for using standard folders
                              # False for user-defined folders listed below
armA_PATH = 'spatial'
armB_PATH = 'angular'

# Time tag output folder?
TT_BOOL = True

#   Definition of folders
DATA_DIR   = ""
OUTPUT_DIR = ""
G2_DIR     = ""

# Cordinates for data cropping
# Na = 250
# Nb = 360

# Binning
#! if apply physical binning (seting on cameras), pixel size is 6.5e-3 * phybin
binA, binB = 4,5
cambinA = 5
pixel = 6.5e-3
# NA, NB     = Na//binA, Nb//binB
pixA, pixB = cambinA * pixel * binA, pixel * binB
# REFOCUSING
focal = 30
MA    = 4.2
MB    = 0.32

def axial(shift):
    position = - focal/(1 + MA/MB * pixB/pixA /shift)
    return position

def shift(position):
    shift = - position * MA/MB / (position + focal) * pixB/pixA
    return shift
# Array of planes to be refocused
