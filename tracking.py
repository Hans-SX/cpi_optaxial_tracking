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

from utils import readConfig, setDirectories_twocams, Calculating_G2, Refocusing_by_Shifting, plot_G2s, target_axials, Measure_Benchmarking, Measures, Timer
from config import axial, shift


exec(readConfig())

parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', type=str, nargs='?', const='refocused')
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
outpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outDir, armAfiles, armBfiles = setDirectories_twocams(stdData=STD_PATH, stdOut=STD_PATH, timeTag=TT_BOOL, dataPath=datapath, outPath=outpath, armA=armA_PATH, armB=armB_PATH)

# shifts = np.linspace(-3, 3, 40)

"""
try_shifts: for each step the shifts are spanned from the expected point, +- 0.2 mm with 10 intervals each direction. Then converted to shifts.
"""
expect_ref = 6.87 - 0.5 * np.array(range(34))
expect_ref = [(expect_ref[i] + expect_ref[i+1])/2 for i in range(len(expect_ref)-1)]
try_ref_to = [target_axials(x, 10, 0.2) for x in expect_ref]
try_shifts = [[shift(try_ref_to[x][y]) for y, _ in enumerate(try_ref_to[x])] for x, _ in enumerate(try_ref_to)]
# try_shifts = try_shifts[-1:] # for only one step

cyc = 0
init_guess = [1, 0, 0, 80]
ref_steps = dict()
axial_steps = dict()
g2s = []
timer = Timer()
timer.start("Whole refocusing")
for Afile, Bfile, shifts in zip(armAfiles, armBfiles, try_shifts):
# for Afile, Bfile in zip(armAfiles, armBfiles):
    timer.start("Refocusing interval " + str(cyc+1))
    arr = Calculating_G2(Afile, Bfile)
    G2 = arr.correlation(binA, binB)
    g2s.append(G2)
    G2 = arr.padding(pad=25)
    
    if not os.path.exists(join(outDir, str(cyc+1))):
        os.makedirs(join(outDir, str(cyc+1)))
    
    refocusing = Refocusing_by_Shifting(G2, shifts, focal, MA, MB, pixA, pixB, join(outDir, str(cyc+1)))
    refocused_results, axial_results = refocusing.evaluate_refocusG2fast_parallel()

    ref_steps["s" + str(cyc+1)] = refocused_results
    axial_steps["s" + str(cyc+1)] = axial_results
    # min_sig_ind, init_guess = next_shift_range(refocused_results, init_guess)
    #? somehow need more tials to fit without change the shifts.
    # refocusing._plt(refocused_results[min_var_ind], shifts[min_var_ind], cyc)

    # shifts = np.linspace(shifts[min_var_ind] - 1, shifts[min_var_ind] + 1, 10)

    if (cyc+1) % 10 == 0:
        print("Step " + str(cyc+1) + " finished.")
    
    timer.stop("Refocusing interval " + str(cyc+1))
    cyc += 1

timer.stop("Whole refocusing")

with open(join(outDir, "ref_steps.pkl"), "wb") as f1:
    pickle.dump(ref_steps, f1)

with open(join(outDir, "axial_steps.pkl"), "wb") as f2:
    pickle.dump(axial_steps, f2)

print("Refocusing done, plot G2s.")
plot_G2s(g2s, outDir)

"""
Measure analysis is separated to another process.

print("Measure analysis.")
measures = Measures()
m1 = Measure_Benchmarking(ref_steps, axial_steps)
m1.save_analysis(measures.apply_measures(apply_measures), expect_ref, outDir)
# m1.save_analysis(measures.apply_measures(), expect_ref[-1], outDir)  # for only one step
"""
# with open(join(outDir, "G2.npy"), "wb") as f3:
#     np.save(f3, g2s)