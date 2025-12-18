#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:58:41 2025

@author: massaro
"""
#%%
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from time import time
from scipy.ndimage import shift as ndi_shift

"""
Original code from Gianlorenzo for refocusing by shifting pixels.
It is splitted and altered to utils.py and tracking.py.
This script is not called in the current working flow.
"""

angular = io.imread("./1h_lowTemp/data/angular/angular.tif")
spatial = io.imread("./1h_lowTemp/data/spatial/spatial.tif")

#%%
def bin_3d_array(arr, bin_size):
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

#%%
Nframes, Na, Nb = len(angular), spatial.shape[1], angular.shape[2]

binA, binB = 4,5
pixel = 6.5e-3
focal = 30
MA    = 4.2
MB    = 0.32

NA, NB     = Na//binA, Nb//binB
pixA, pixB = pixel * binA, pixel * binB

spatial    = bin_3d_array(spatial, binA).reshape(Nframes, NA*NA).astype('float32')
angular    = bin_3d_array(angular, binB).reshape(Nframes, NB*NB).astype('float32')

G2 = np.matmul(spatial.T, angular) / np.tensordot(
    np.sum(spatial,0),
    np.mean(angular,0)
    , axes=0) - 1
G2 = G2.reshape((NA, NA, NB, NB))
#%%
# Refocus


min_refoc     = -10
max_refoc     = +10
TO_REFOCUS    = np.arange(min_refoc,max_refoc)

#%%

def shift_with_zeros(arr, shift):
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
#%%
def shift_subpixel(arr, shift):
    floored = np.floor(shift).astype(int)
    remaind_y, remaind_x = shift - floored  # Extract fractional shifts

    tmp = shift_with_zeros(arr, floored)  # Apply integer shift

    # Compute weighted sums using np.roll instead of np.pad
    tmp_y  = np.roll(tmp, -1, axis=0) if remaind_y else tmp
    tmp_x  = np.roll(tmp, -1, axis=1) if remaind_x else tmp
    tmp_xy = np.roll(tmp_y, -1, axis=1) if remaind_x else tmp_y

    # Interpolation using bilinear weighting
    return (1 - remaind_y) * (1 - remaind_x) * tmp + \
           (1 - remaind_y) * remaind_x * tmp_x + \
           remaind_y * (1 - remaind_x) * tmp_y + \
           remaind_y * remaind_x * tmp_xy
#%%
def refocusG2fast(array4D, shift):
    """
    Refocuses a 4D correlation array by shifting each 2D slice and summing the result.

    Parameters:
    array4D (numpy.ndarray): Input array of shape (N, M, Ny, Nx).
    shift (int): Shift scaling factor.

    Returns:
    numpy.ndarray: Refocused 2D array of shape (N, M).
    """
    N, M, NBy, NBx = array4D.shape
    y_shifts = shift * (np.arange(NBy) - NBy // 2)  # Compute all y shifts
    x_shifts = shift * (np.arange(NBx) - NBx // 2)  # Compute all x shifts

    refocused = np.zeros((N, M), dtype=array4D.dtype)  # Initialize result

    # Apply shifts efficiently using broadcasting
    for j in range(NBy):
        for i in range(NBx):
            if -y_shifts[j]%1==0 and x_shifts[i]%1==0:
                refocused += shift_with_zeros(array4D[:, :, j, i], (-y_shifts[j], x_shifts[i]))
                # refocused += shift_with_zeros(array4D[:, :, j, i], (-y_shifts[j], -x_shifts[i]))  # for simulaiton data

            else:
                refocused += shift_subpixel(array4D[:, :, j, i], (-y_shifts[j], x_shifts[i]))
                # refocused += shift_subpixel(array4D[:, :, j, i], (-y_shifts[j], -x_shifts[i])) # for simulation data
    axial = -focal/(1 + MA/MB * pixB/pixA /shift) if shift!=0 else 0
   
    return refocused, axial
#%%
import numpy as np
from joblib import Parallel, delayed
import os
# os.chdir(os.path.expanduser("~/Desktop"))  # Change this to your actual working directory

def evaluate_refocusG2fast_parallel(array4D, shifts, n_jobs=-1):
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
    shifts = np.asarray(shifts)  # Ensure shifts is a NumPy array
    num_shifts = len(shifts)
    N, M, _, _ = array4D.shape  # Get output shape

    # Run refocusG2fast in parallel for each shift
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(refocusG2fast)(array4D, shift) for shift in shifts
    )

    # Extract results
    refocused_results = np.array([res[0] for res in results])  # Stack refocused outputs
    axial_results = np.array([res[1] for res in results])      # Extract axial outputs

    return refocused_results, axial_results
#%%
padding = 25
G2 = np.pad(G2, ((padding,padding),(padding,padding),(0,0),(0,0)))

start = time()
shifts = np.linspace(-3, 3, 40)  # Example shift values
refocused_arr, axial_arr = evaluate_refocusG2fast_parallel(G2, shifts)
# correc_arr   , axial_arr = evaluate_refocusG2fast_parallel(np.zeros_like(G2)+1, shifts)
print(time()-start)
# refocused_arr /= correc_arr



# plt.imshow(ref); plt.show()
# plt.plot(range(NA), np.sum(ref,0), range(NA), np.sum(ref,1)); plt.show()