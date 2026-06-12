import argparse
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from utils import readConfig, setDirectories_twocams, Calculating_G2

exec(readConfig())

parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', nargs='?', default='refocused', type=str)
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
outpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outDir, armAfiles, armBfiles = setDirectories_twocams(stdData=STD_PATH, stdOut=STD_PATH, timeTag=TT_BOOL, dataPath=datapath, outPath=outpath, armA=armA_PATH, armB=armB_PATH)

save_path = join(outpath, 'corr')
if not os.path.exists(save_path):
    os.makedirs(save_path)
cyc = 0
maxG2 = list()
minG2 = list()
for Afile, Bfile in zip(armAfiles, armBfiles):
# for Afile, Bfile in zip(armAfiles, armBfiles):
    arr = Calculating_G2(Afile, Bfile, frames=100)
    G2 = arr.correlation(binA, binB)

    maxG2.append(np.max(np.sum(G2, axis=(1,3))))
    minG2.append(np.min(np.sum(G2, axis=(1,3))))
    fig, (ax1,ax2) = plt.subplots(2,figsize=(6,10))
    im1 = ax1.imshow(np.sum(G2, axis=(1,3)), cmap="gray")
    im2 = ax2.imshow(np.sum(G2, axis=(0,2)), cmap="gray")
    ax1.set_title("xA-xB Corr. Func."); ax2.set_title("yA-yB Corr. Func.")
    fig.colorbar(im1, ax=ax1); fig.colorbar(im2, ax=ax2)
    fig.savefig(join(save_path, f"G2_{cyc+1:03d}"))
    cyc += 1
    plt.close("all")

np.savez(join(save_path, "G2x_max_min.npz"), maxg2=maxG2, ming2=minG2)

