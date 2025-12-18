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
import pickle
from joblib import Parallel, delayed
from functools import partial
import decimal



def plot_G2s(g2s, fpath):
    if not os.path.exists(joinDir(fpath, 'G2s')):
        os.makedirs(joinDir(fpath, 'G2s'))
    cyc = 0
    for g2 in g2s:
        fig, (ax1,ax2) = plt.subplots(2,figsize=(6,10))
        im1 = ax1.imshow(np.sum(g2, axis=(1,3)), cmap="gray")
        im2 = ax2.imshow(np.sum(g2, axis=(0,2)), cmap="gray")
        ax1.set_title("xA-xB Corr. Func."); ax2.set_title("yA-yB Corr. Func.")
        fig.colorbar(im1, ax=ax1); fig.colorbar(im2, ax=ax2)
        fig.savefig(joinDir(fpath, "G2s", f"G2_{cyc+1:03d}"))
        cyc += 1
        plt.close("all")

"""
It takes time to compute, and it may return a random spot as best in a noisy one.
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
"""
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

class Calculating_G2():
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
        # NA = self.fileA.shape[1]//binA
        # NB = self.fileB.shape[1]//binB
        NA1 = self.fileA.shape[1]//binA
        NB1 = self.fileB.shape[1]//binB
        NA2 = self.fileA.shape[2]//binA
        NB2 = self.fileB.shape[2]//binB
        spatial = self._bin_3d_array(self.fileA, binA).reshape(N, NA1*NA2).astype("float32")
        angular = self._bin_3d_array(self.fileB, binB).reshape(N, NB1*NB2).astype("float32")

        # self.G2 = np.matmul(spatial.T, angular) / np.tensordot(np.sum(spatial,0), np.mean(angular,0), axes=0) - 1
        self.G2 = np.matmul(spatial.T, angular)/N - np.tensordot(np.mean(spatial,0), np.mean(angular,0), axes=0)
        self.G2 = self.G2.reshape(NA1, NA2, NB1, NB2)
        return self.G2
    
    def padding(self, pad):
        self.G2 = np.pad(self.G2, ((pad, pad), (pad, pad), (0,0), (0,0)))
        return self.G2

def target_axials(x, n, stepsize):
    """
    x: the expected position
    n: 2*n + 1 will be the total examined positions
    """
    left_values = [x - i * stepsize for i in range(1, n+1)]
    right_values = [x + i * stepsize for i in range(1, n+1)]
    return left_values[::-1] + [x] + right_values

class Refocusing_by_Shifting():
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
        plt.imshow(refocused_result, cmap='gray')
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
                    # for simulation data, reflect x.
                    # refocused += self._shift_with_zeros(self.array4D[:, :, j, i], (-y_shifts[j], -x_shifts[i]))
                    refocused += self._shift_with_zeros(self.array4D[:, :, j, i], (-y_shifts[j], x_shifts[i]))
                else:
                    # refocused += self._shift_with_zeros(self.array4D[:, :, j, i], (-y_shifts[j], -x_shifts[i]))
                    refocused += self._shift_with_zeros(self.array4D[:, :, j, i], (-y_shifts[j], x_shifts[i]))
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

class Signal_to_BG_Noise_Ratio():
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def _rescaling(self, arr, rescale=(1, 1e1)):
        # Rescaling the values of an 1-D array.
        return rescale[0] + (rescale[1] - rescale[0]) * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    def _rescaling_pixelwise(self, arr, rescale=(1, 1e1)):
        # Rescaling the values of each image.
        
        return rescale[0] + (rescale[1] - rescale[0]) * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    #def sbnr(self, threshold, ref_step, axial_exp):
    def sbnr(self, ref_step, axial_exp, threshold=0.005):
        background = np.asarray([ref_step[shift] * (ref_step[shift] < threshold *np.max(ref_step[shift])) for shift in range(len(ref_step))])
        background = np.std(background, axis=(1,2))
        re_bg = self._rescaling(background)
        signal = np.asarray([ref_step[shift] * (ref_step[shift] >= 0.9 *np.max(ref_step[shift])) for shift in range(len(ref_step))])
        signal = np.mean(ref_step, axis=(1,2))
        re_sig = self._rescaling(signal)
        sbnr = re_sig / re_bg
        # sbnr = signal / background
        sbnr_exp = sbnr[axial_exp]
        if len(sbnr) < 2:
            ind_best = 0
        else:
            ind_best = np.argwhere(sbnr == np.max(sbnr))[0]
        # name = "sbnr_" + f'%.1E' % decimal.Decimal(threshold)
        name = 'SNR, '
        return np.max(sbnr), ind_best, sbnr_exp, name
    
    def apply(self):
        measures = []
        for threshold in self.thresholds:
            measures.append(partial(self.sbnr, threshold=threshold))
        return measures
    
class Measures(Signal_to_BG_Noise_Ratio):
    """
    return:
        val_best: the best value in a given step of the given measure.
        ind_best: the index of the corresponding best value, it is connected to the axial position.
        val_exp: the value of the given measure on the expected position of the given step.
        str: return the measure name for the figure.
    """
    def __init__(self):
        pass

    def max_intensity(self, ref_step, axial_exp):
        val_best = np.max(ref_step, axis=(1,2))
        val_exp = val_best[axial_exp]
        ind_best = np.argwhere(val_best == np.max(val_best))
        val_best = np.max(val_best)
        return val_best, ind_best, val_exp, "max_intensity"

    def min_std(self, ref_step, axial_exp):
        val_best = np.std(ref_step, axis=(1,2))
        val_exp = val_best[axial_exp]
        ind_best = np.argwhere(val_best == np.min(val_best))
        val_best = np.min(val_best)
        return val_best, ind_best, val_exp, "min_std"

    def min_std_background(self, ref_step, axial_exp, threshold=0.005):
        background = np.asarray([ref_step[shift] * (ref_step[shift] < threshold *np.max(ref_step[shift])) for shift in range(len(ref_step))])
        val_best = np.std(background, axis=(1,2))
        val_exp = val_best[axial_exp]
        ind_best = np.argwhere(val_best == np.min(val_best))
        val_best = np.min(val_best)
        name = "min_std_bg_" + f'%.1E' % decimal.Decimal(threshold)
        return val_best, ind_best, val_exp, name
    
    def min_abs_min_intensity(self, ref_step, axial_exp):
        val_best = np.abs(np.min(ref_step, axis=(1,2)))
        val_exp = val_best[axial_exp]
        ind_best = np.argwhere(val_best == np.min(val_best))
        val_best = np.min(val_best)
        return val_best, ind_best, val_exp, "min_abs_min_intensity"
    
    def maxint_minstdbg(self, ref_step, axial_exp, threshold=0.005):
        background = np.asarray([ref_step[shift] * (ref_step[shift] < threshold *np.max(ref_step[shift])) for shift in range(len(ref_step))])
        rank_maxint = np.argsort(np.max(ref_step, axis=(1,2)))
        rank_minstdbg = np.argsort(-1 * np.std(background, axis=(1,2)))
        rank = rank_maxint + rank_minstdbg
        rank_exp = rank[axial_exp]
        ind = np.argwhere(rank == np.max(rank))[0]  #Don't know why, it is in [[]] form.
        rank = np.max(rank)
        # name = "maxint_minstdbg_" + f'%.1E' % decimal.Decimal(threshold)
        name = 'SRSN, '
        return rank, ind, rank_exp, name

    def apply_measures(self, apply_measures):
        ms = []
        for mstr in apply_measures:
            m = getattr(self, mstr, None)
            if callable(m):
                ms.append(m)
            else:
                raise AttributeError(f"Measure '{m}' not found.")
        return ms
    
class STD_BG_Thresholds():
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def min_std_background(self, threshold, ref_step, axial_exp):
        background = np.asarray([ref_step[shift] * (ref_step[shift] < threshold *np.max(ref_step[shift])) for shift in range(len(ref_step))])
        val_best = np.std(background, axis=(1,2))
        val_exp = val_best[axial_exp]
        ind_best = np.argwhere(val_best == np.min(val_best))
        val_best = np.min(val_best)
        name = "min_std_bg_" + f'%.1E' % decimal.Decimal(threshold)
        return val_best, ind_best, val_exp, name
    
    def maxint_minstdbg(self, threshold, ref_step, axial_exp):
        background = np.asarray([ref_step[shift] * (ref_step[shift] < threshold *np.max(ref_step[shift])) for shift in range(len(ref_step))])
        rank_maxint = np.argsort(np.max(ref_step, axis=(1,2)))
        rank_minstdbg = np.argsort(-1 * np.std(background, axis=(1,2)))
        rank = rank_maxint + rank_minstdbg
        rank_exp = rank[axial_exp]
        ind = np.argwhere(rank == np.max(rank))[0]  #Don't know why, it is in [[]] form.
        rank = np.max(rank)
        name = "maxint_minstdbg_" + f'%.1E' % decimal.Decimal(threshold)
        return rank, ind, rank_exp, name
    
    def apply_std_bg(self):
        measures = []
        for threshold in self.thresholds:
            measures.append(partial(self.min_std_background, threshold))
        return measures
    
    def apply_maxint_minstdbg(self):
        measures = []
        for threshold in self.thresholds:
            measures.append(partial(self.maxint_minstdbg, threshold))
        return measures

class Measure_Benchmarking():
    """
    ref_steps: the refocused results in different steps, images.
    axials_steps: the axial positions corresponding to the refocused results, axial positions.
    measure: to decide which is the best refocused.

    return:
        val_bests: the best values in all steps of the given measure.
        axial_bests: the axial positions corresponding to the best values.
        vals_exp: the values of the given measure on the expected positions of all steps.
    """
    def __init__(self, ref_steps, axials_steps):
        self.ref = ref_steps
        self.axials = axials_steps
        # The expected position is the middle one, the refocused range is setup accordingly.
        self.expected = (len(ref_steps['s1']) - 1 )// 2

    def _measure_vals_axials(self, measure):
        vals_exp  = []
        val_bests = []
        axial_bests = []
        for step in self.ref:
            val_best, ind_best, val_exp, measure_name = measure(self.ref[step], self.expected)
            val_bests.append(val_best)
            axial_bests.append(self.axials[step][ind_best])
            vals_exp.append(val_exp)
        axial_bests = np.asarray(axial_bests).reshape(-1)
        return val_bests, axial_bests,  vals_exp, measure_name
    
    def total_diff(self, axial_exp, axial_bests):
        return np.sum(np.abs(axial_exp - axial_bests))

    def save_analysis(self, measures, axial_exp, path):
        fig = plt.figure()
        plt.plot(range(1, 1 + len(axial_exp)), axial_exp, label='Expected position', c='black')
        axial_estimate = dict()

        for measure in measures:
            val_bests, axial_bests,  val_exp, measure_name = self._measure_vals_axials(measure)
            axial_estimate[measure_name] = axial_bests
            mae = self.total_diff(axial_exp, axial_bests) / len(axial_bests)  # Mean Absolute Error
            fro = np.linalg.norm(axial_bests - axial_exp) # Frobenius-norm / num of expected positions.
            mse = np.mean((axial_bests - axial_exp)**2)  # Mean Squared Error
            # fname = joinDir(path, measure_name)
            plt.scatter(range(1, 1 + len(axial_exp)), axial_bests, label= measure_name + ' (MSE, MAE) = ' + f'({mse:.3f}, {mae:.3f})')
            # plt.plot(range(1, 1 + len(axial_exp)), axial_bests, label='Ref., ' + measure_name + '_' + f'{evalu:.3f}')
            # self._plot_performance(fname)
            
            # save_data = {
            #     'val_bests': val_bests,
            #     'axial_bests': axial_bests,
            #     'val_exp': val_exp,
            #     'axial_exp': axial_exp
            # }

            # with open(path + "/" + measure_name + ".pkl", "wb") as f1:
            #     pickle.dump(save_data, f1)

        plt.xlabel("Every 100 frames.")
        plt.ylabel("Axial position in mm.")
        plt.legend()
        fig.savefig(path + '/measures_comparison_scatter', dpi='figure', transparent=False)
        plt.close(fig)
        return axial_estimate
    
class Timer:
    def __init__(self):
            self.start_times = {}
            self.elapsed_times = {}
    def start(self, label):
        #   self.start_times[label] = time.time()
          self.start_times[label] = time.perf_counter()
    def stop(self, label):
        if label in self.start_times:
            # elapsed_time = time.time() - self.start_times[label]
            elapsed_time = time.perf_counter() - self.start_times[label]
            self.elapsed_times[label] = elapsed_time
        else:
             print(f"Timer for {label} is not started.")
    def savefile(self, filename):
        with open(filename, 'w') as file:
            for label in self.elapsed_times:
                file.write(f"{label}, {self.elapsed_times[label]}\n")