"""
Created on  Mar. 2025

@author: Shih-Xian
"""
import numpy as np

import argparse
import os
from os.path import join
import matplotlib.pyplot as plt
import pickle

from utils import readConfig, setDirectories_twocams, calculating_G2, refocusing_by_shifting, next_shift_range
from config import shift, axial

exec(readConfig())

parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', type=str, nargs='?', const='')
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
outpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outDir, armAfiles, armBfiles = setDirectories_twocams(stdData=STD_PATH, stdOut=STD_PATH, timeTag=TT_BOOL, dataPath=datapath, outPath=outpath, armA=armA_PATH, armB=armB_PATH)
fig_path = outDir

# shifts = np.linspace(-3, 3, 40)
idea_ref = 6.87 - 0.5 * np.array(range(1, 34))
try_ref_to = [np.linspace(x, x + 6, 2) for x in idea_ref]
# try_ref_to = [np.linspace(x - 0.5, x + 0.5, 5) for x in idea_ref]
try_shifts = [[shift(try_ref_to[x][y]) for y, _ in enumerate(try_ref_to[x])] for x, _ in enumerate(try_ref_to)]

cyc = 0
init_guess = [1, 0, 0, 80]
ref_steps = dict()
axial_steps = dict()
g2s = []
for Afile, Bfile, shifts in zip(armAfiles, armBfiles, try_shifts):
    arr = calculating_G2(Afile, Bfile)
    G2 = arr.correlation(binA, binB)
    g2s.append(G2)
    G2 = arr.padding(pad=25)
    
    if not os.path.exists(join(fig_path, str(cyc+1))):
        os.makedirs(join(fig_path, str(cyc+1)))
    
    refocusing = refocusing_by_shifting(G2, shifts, focal, MA, MB, pixA, pixB, join(fig_path, str(cyc+1)))
    refocused_results, axial_results = refocusing.evaluate_refocusG2fast_parallel()

    ref_steps["s" + str(cyc+1)] = refocused_results
    axial_steps["s" + str(cyc+1)] = axial_results
    # min_sig_ind, init_guess = next_shift_range(refocused_results, init_guess)
    #? somehow need more tials to fit without change the shifts.
    # refocusing._plt(refocused_results[min_var_ind], shifts[min_var_ind], cyc)

    # shifts = np.linspace(shifts[min_var_ind] - 1, shifts[min_var_ind] + 1, 10)

    if (cyc+1) % 10 == 0:
        print("Step " + str(cyc+1) + " finished.")
    
    cyc += 1

with open(join(fig_path, "ref_steps.pkl"), "wb") as f1:
    pickle.dump(ref_steps, f1)

with open(join(fig_path, "axial_steps.pkl"), "wb") as f2:
    pickle.dump(axial_steps, f2)

with open(join(fig_path, "G2.npy"), "wb") as f3:
    np.save(f3, g2s)