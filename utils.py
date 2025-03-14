"""
Created on  Mar. 03, 2025

@author: Shih-Xian, Gianlorenzo (Those are from CPI.py.)
"""

import numpy as np
from scipy.optimize import curve_fit
from tifffile import imread
from skimage import io
from os.path import join as joinDir
from datetime import datetime
import os, sys, time
import matplotlib.pyplot as plt
from os.path import join as joinDir
import shutil
from joblib import Parallel, delayed



# class plot_ref():
#     def __init__(self):
    
#     def _plt(self, refocused_result, axial_result, path):
#         path = joinDir(path, "_Refocus_plot_"+ str(round(axial_result, 3))+".png")

#         fig = plt.figure()
#         plt.imshow(refocused_result)
#         plt.title("Refocused image_oof= "+ str(round(axial_result, 3)) + " mm")
#         plt.colorbar()
#         fig.savefig(path, dpi='figure',transparent=False)
#         plt.close("all")

#     def plt_parallel(self, refocused_results, axial_results, path, n_jobs=-1):
#         Parallel(n_jobs=n_jobs, backend="loky")(delayed(self._plt)(refocused_result, axial_result, path) for refocused_result , axial_result in refocused_results, axial_results)

def gaussian2D(xy, amp, x0, y0, std):
    x, y = xy
    g2D = amp * np.exp(-((x - x0)**2/std**2 + (y - y0)**2/std**2)/2)
    return g2D.ravel()

def next_shift_range(refocused_arr, init_guess):
    stds = []
    x = np.arange(refocused_arr.shape[1]) - refocused_arr.shape[1]//2
    y = np.arange(refocused_arr.shape[2]) - refocused_arr.shape[2]//2
    x, y = np.meshgrid(x, y)
    
    bounds = ([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])
    for i in range(refocused_arr.shape[0]):
        (amp, x0, y0, std), _ = curve_fit(gaussian2D, (x, y), refocused_arr[i].ravel(), p0=init_guess, bounds=bounds, maxfev=2000)
        stds.append(std)
    min_var_ind = np.argmin(np.array(stds))
    next_guess = [amp, x0, y0, std]
    return min_var_ind, next_guess

# ---------------------------------------------------------------------------------

def PrintSectionInit(text):
    '''
    Prints section name "text" with an indent

    Parameters
    ----------
    text : string

    Returns
    -------
    None.

    '''
    print("{:>17}>> {}".format("\n_______________", text))

def PrintSectionClose():
    '''
    Prints "DONE" at the end of a section

    Returns
    -------
    None.

    '''
    print("{:>17}{}".format("", 'DONE.'))

def Print(label, text):
    '''
    Prints "label": "text"

    Parameters
    ----------
    label : string or number.
    text :  string or number.

    Returns
    -------
    None.

    '''
    print("{:>15}: {}".format(label, text))

def FilePath():
    '''
    Returns the directory of the current python script
    '''
    return os.path.dirname(os.path.abspath(__file__))

def DataDir(std=True, path=None):
    '''
    Returns the output directory. If std is True, the default directory is 
    given returned. If std is not True, the variable 'path' is returned.

    Parameters
    ----------
    std : TYPE, Boolean
        DESCRIPTION. The default is True.
    path : TYPE, String
        DESCRIPTION. The default is None.

    Returns
    -------
    path : string.

    '''
    if std:
        path = joinDir(FilePath(), os.pardir,'data')    
    return path

def OutDir(timeTag=False, std=True, path=None):
    '''
    Returns the output directory. If std is True, the default directory is 
    returned. If std is not True, the variable 'path' is returned. If timeTag
    is True, current date and time is added to the output folder name

    Parameters
    ----------
    timeTag : TYPE, Boolean
        DESCRIPTION. The default is False.
    std : TYPE, Boolen
        DESCRIPTION. The default is True.
    path : TYPE, String
        DESCRIPTION. The default is None.

    Returns
    -------
    path : TYPE
        DESCRIPTION.

    '''
    if std:
        path = joinDir(FilePath(), os.pardir,'output')
    if timeTag:
        path += '_'+Now()
    return path

def Now():
    return datetime.now().strftime("%Y-%m-%d %H.%M")

def readConfig():
    PrintSectionInit('Reading parameters from config...')
    file = open("config.py")
    cfg  = file.read()
    file.close()
    print(cfg)
    PrintSectionClose()
    return cfg

def setDirectories_twocams(stdData=True, stdOut=True, timeTag=False, 
                   dataPath=None, outPath=None, armA='spatial', armB='angular'):
    '''
    Creates the folder of interest and returns the output directory and a list
    of files for data reading.

    Parameters
    ----------
    stdData : TYPE, Boole
        DESCRIPTION. The default is True.
    stdOut : TYPE, Boole
        DESCRIPTION. The default is True.
    timeTag : TYPE, Boole
        DESCRIPTION. The default is False.
    dataPath : TYPE, String
        DESCRIPTION. The default is None.
    outPath : TYPE, String
        DESCRIPTION. The default is None.

    Returns
    -------
    outPath : String.
    filename : Array of files to read.

    '''
    PrintSectionInit('Setting up folders...')
    
    dataPath = DataDir(stdData, dataPath)
    armAPath = joinDir(dataPath, armA)
    armBPath = joinDir(dataPath, armB)
    outPath  = OutDir(timeTag, stdOut, outPath)
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    shutil.copy("./config.py", joinDir(outPath, "config.py"))
    armAfilename  = os.listdir(armAPath)
    armAfilename  = [os.path.join(armAPath, armAfilename[i]) 
                 for i in range(len(armAfilename))]
    armBfilename  = os.listdir(armBPath)
    armBfilename  = [os.path.join(armBPath, armBfilename[i]) 
                 for i in range(len(armBfilename))]
    Print("[DATA PATH]", dataPath)
    Print("[ARM A PATH]", armAPath)
    Print("[ARM B PATH]", armBPath)
    Print("[OUT PATH]", outPath)
    
    PrintSectionClose()
    return outPath, armAfilename, armBfilename

def PlotCorrFunc2D(corrFunc, outDir, differential=False):
    if not differential:
        PrintSectionInit("Printing 2D corr. function")
    else:
        PrintSectionInit("Printing 2D differential corr. function")
    corrDir = joinDir(outDir,"Correlation function")
    if not os.path.exists(corrDir): 
        os.makedirs(corrDir)
    fig, (ax1,ax2) = plt.subplots(2,figsize=(20,20))
    im1 = ax1.imshow(np.sum(corrFunc, axis=(1,3)), cmap="gray")
    im2 = ax2.imshow(np.sum(corrFunc, axis=(0,2)), cmap="gray")
    ax1.set_title("xA-xB Corr. Func."); ax2.set_title("yA-yB Corr. Func.")
    fig.colorbar(im1, ax=ax1); fig.colorbar(im2, ax=ax2)  
    if differential: 
        fig.savefig(joinDir(corrDir, "Corr_func_diff.tif"),
                dpi='figure',transparent=False)
    else: 
        fig.savefig(joinDir(corrDir, "Corr_func.tif"),
                dpi='figure',transparent=False)
    plt.close("all")
    PrintSectionClose()

class calculating_G2():
    def __init__(self, fileA, fileB):
        self.fileA = imread(fileA)
        self.fileB = imread(fileB)

    def _bin_3d_array(self, arr, bin_size):
        """
        Bins a 3D NumPy array along the last two axes while preserving the first axis.

        Parameters:
        arr (numpy.ndarray): Input 3D array of shape (N, H, W).
        bin_size (int): The factor by which to downsample the last two dimensions.

        Returns:
        numpy.ndarray: Binned array of shape (N, H//bin_size, W//bin_size).
        """
        if arr.ndim != 3:
            raise ValueError("Input array must be 3D.")
    
        N, H, W = arr.shape
        if H % bin_size != 0 or W % bin_size != 0:
            arr = arr[:, H % bin_size:, W % bin_size:]

        # Reshape and sum over the new axes
        arr_binned = arr.reshape(N, H//bin_size, bin_size, W//bin_size, bin_size).sum(axis=(2, 4))
    
        return arr_binned
    
    def correlation(self, binA, binB):
        N = self.fileA.shape[0]
        NA = self.fileA.shape[1]//binA
        NB = self.fileB.shape[1]//binB
        spatial = self._bin_3d_array(self.fileA, binA).reshape(N, NA*NA).astype("float32")
        angular = self._bin_3d_array(self.fileB, binB).reshape(N, NB*NB).astype("float32")

        # self.G2 = np.matmul(spatial.T, angular) / np.tensordot(np.sum(spatial,0), np.mean(angular,0), axes=0) - 1
        # self.G2 = self.G2.reshape((NA, NA, NB, NB))
        self.G2 = np.matmul(spatial.T, angular)/N - np.tensordot(np.mean(spatial,0), np.mean(angular,0), axes=0)
        self.G2 = self.G2.reshape((NA, NA, NB, NB))
        return self.G2
    
    def padding(self, pad):
        self.G2 = np.pad(self.G2, ((pad, pad), (pad, pad), (0,0), (0,0)))
        return self.G2

class refocusing_by_shifting():
    def __init__(self, array4D, shifts, focal, MA, MB, pixA, pixB, path):
        self.array4D = array4D
        self.shifts = shifts
        self.focal, self.MA, self.MB, self.pixA, self.pixB = focal, MA, MB, pixA, pixB
        self.path = path
        self.axial = 0
    
    def _plt(self, refocused_result, shift, ind):
        path = joinDir(self.path, str(ind) + "_Refocus_plot_"+ str(round(shift, 3))+".png")

        self.axial = self._axial(shift)
        fig = plt.figure()
        plt.imshow(refocused_result)
        plt.title("Refocused image_oof = "+ str(round(self.axial, 3)) + " mm")
        plt.colorbar()
        fig.savefig(path, dpi='figure',transparent=False)
        plt.close("all")
    
    def _axial(self, shift):
        self.axial = -self.focal/(1 + self.MA/self.MB * self.pixB/self.pixA /shift) if shift!=0 else 0
        return self.axial
    
    def _shift_with_zeros(self, arr, shift):
        """
        Shifts a NumPy array along both axes (last two dimensions), filling rolled-over values with zeros.

        Parameters:
        arr (numpy.ndarray): Input array of shape (..., H, W).
        shift (tuple): (shift_y, shift_x)

        Returns:
        numpy.ndarray: Shifted array with zeros filling rolled-over positions.
        """
        shift_y, shift_x = int(shift[0]), int(shift[1])  # Ensure integer shifts
        arr_shifted = np.zeros_like(arr)

        # Compute valid index ranges (force integer)
        src_y_start = max(0, -shift_y)
        src_y_end = arr.shape[-2] - max(0, shift_y)
        src_x_start = max(0, -shift_x)
        src_x_end = arr.shape[-1] - max(0, shift_x)

        dest_y_start = max(0, shift_y)
        dest_y_end = arr.shape[-2] - max(0, -shift_y)
        dest_x_start = max(0, shift_x)
        dest_x_end = arr.shape[-1] - max(0, -shift_x)

        # Prevent empty slice assignment
        if dest_y_start < dest_y_end and dest_x_start < dest_x_end:
            arr_shifted[..., dest_y_start:dest_y_end, dest_x_start:dest_x_end] = \
                arr[..., src_y_start:src_y_end, src_x_start:src_x_end]

        return arr_shifted

    def _shift_subpixel(self, arr, shift):
        floored = np.floor(shift).astype(int)
        remaind_y, remaind_x = shift - floored  # Extract fractional shifts

        tmp = self._shift_with_zeros(arr, floored)  # Apply integer shift

        # Compute weighted sums using np.roll instead of np.pad
        tmp_y  = np.roll(tmp, -1, axis=0) if remaind_y else tmp
        tmp_x  = np.roll(tmp, -1, axis=1) if remaind_x else tmp
        tmp_xy = np.roll(tmp_y, -1, axis=1) if remaind_x else tmp_y

        # Interpolation using bilinear weighting
        return (1 - remaind_y) * (1 - remaind_x) * tmp + \
            (1 - remaind_y) * remaind_x * tmp_x + \
            remaind_y * (1 - remaind_x) * tmp_y + \
            remaind_y * remaind_x * tmp_xy

    def _refocusG2fast(self, shift, ind):
        """
        Refocuses a 4D correlation array by shifting each 2D slice and summing the result.

        Parameters:
        array4D (numpy.ndarray): Input array of shape (N, M, Ny, Nx).
        shift (int): Shift scaling factor.

        Returns:
        numpy.ndarray: Refocused 2D array of shape (N, M).
        """
        N, M, NBy, NBx = self.array4D.shape
        y_shifts = shift * (np.arange(NBy) - NBy // 2)  # Compute all y shifts
        x_shifts = shift * (np.arange(NBx) - NBx // 2)  # Compute all x shifts

        refocused = np.zeros((N, M), dtype=self.array4D.dtype)  # Initialize result

        # Apply shifts efficiently using broadcasting
        for j in range(NBy):
            for i in range(NBx):
                if -y_shifts[j]%1==0 and x_shifts[i]%1==0:
                    refocused += self._shift_with_zeros(self.array4D[:, :, j, i], (-y_shifts[j], x_shifts[i]))
                else:
                    refocused += self._shift_subpixel(self.array4D[:, :, j, i], (-y_shifts[j], x_shifts[i]))
        self.axial = self._axial(shift)
        self._plt(refocused, shift, ind)
        return refocused, self.axial

    def evaluate_refocusG2fast_parallel(self, n_jobs=-1):
        """
        Evaluates refocusG2fast in parallel over multiple shift values.

        Parameters:
        - array4D (numpy.ndarray): Input 4D array (N, M, Ny, Nx)
        - shifts (numpy.ndarray or list): Array of shift values
        - n_jobs (int): Number of parallel jobs (-1 uses all available cores)

        Returns:
        - refocused_results (numpy.ndarray): 3D array of shape (len(shifts), N, M)
        - axial_results (numpy.ndarray): 1D array of shape (len(shifts),)
        """
        shifts = np.asarray(self.shifts)  # Ensure shifts is a NumPy array

        # Run refocusG2fast in parallel for each shift
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._refocusG2fast)(shift, ind+1) for ind, shift in enumerate(shifts)
        )

        # Extract results
        refocused_results = np.array([res[0] for res in results])  # Stack refocused outputs
        axial_results = np.array([res[1] for res in results])      # Extract axial outputs

        return refocused_results, axial_results
