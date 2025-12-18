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
binA, binB = 1, 1
cambinA = 10
cambinB = 2
# pixel size of Andor Zyla 5.5 camera.
pixel = 6.5e-3
# pixel size of simulated pinhole
# FoV = 3e-3
# n_pixels = 1024
# pixel = FoV / n_pixels
# NA, NB     = Na//binA, Nb//binB
pixA, pixB = cambinA * pixel * binA, cambinB * pixel * binB
# REFOCUSING
focal = 30
MA    = 4.2    # 1 for simulation data, 4.2 for experiment.
MB    = 0.32    # 1 for simulation data, 0.32 for experiment.


def shift(position):
    shift = - position * MA/MB / (position + focal) * pixB/pixA # for the simulation data: position * MA/MB / (position + focal) * pixB/pixA
    return shift

# apply_measures = ['max_intensity', 'min_std_background', 'maxint_minstdbg']
