#!/usr/bin/env python
# coding: utf-8

# ## Variable binning
# 
# Notebook to make some tests of a variable binning scheme for the background model fit.
# 
# Author: Toby Dixon (toby.dixon.23@ucl.ac.uk)

import re
import pandas as pd
import uproot
import copy
import hist
import math
from legend_plot_style import LEGENDPlotStyle as lps
from datetime import datetime, timezone
from scipy.stats import poisson
from scipy.stats import norm
lps.use('legend')
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
from hist import Hist
import json
from legendmeta import LegendMetadata
import warnings
from iminuit import Minuit, cost
from scipy.stats import expon
from scipy.stats import truncnorm
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit
import utils

vset = tc.tol_cset('vibrant')



# #### Binning algorithm
# 1. Take a list of gamma lines and place each in a single bin
#     * Check there are no overlaps
# 2. Fill in the rest of the space with the continium binning 
#     * Walk through the spectrum, if a new bin can be placed do it.
#     * If distance to next bin < cont_binning/2 append this to the previous bin
#     * If distance to next bin < cont_binning and > cont_binning/2 create a new bin



def insert_bin(bin_edge,bins):
    """ Add a bin edge to a sorted list (bin edges)"""
    insert_index = np.searchsorted(bins, bin_edge)
    sorted_array = np.insert(bins, insert_index, bin_edge)
    return sorted_array
def remove_bin(bin,bins):
    return bins[bins!=bin]


### a list of gamma lines


## start with a basic list of gamma lines
gamma_energies= np.array([583,609,911,1460,1525,1764,2204,2615])
low_energy=565
high_energy=4000
gamma_binning=10
cont_binning=15

bin_edges = np.array([low_energy,high_energy])
print(bin_edges)

for i in range(len(gamma_energies)):

    ### check first for overlaps
    energy = gamma_energies[i]
    if (i!=len(gamma_energies)-1):

        if (energy+gamma_binning/2>gamma_energies[i+1]-gamma_binning/2):
            raise ValueError("Error: Gamma lines are too close - revise")
    bin_edges=insert_bin(energy-gamma_binning/2,bin_edges)
    bin_edges=insert_bin(energy+gamma_binning/2,bin_edges)


bin_edges_only_gamma=bin_edges

### fill in the continuum
for i in range(len(bin_edges_only_gamma)-1):

    ### skip the first part of the gamma line pairs
    if (i%2==1):
        continue
    energy = bin_edges_only_gamma[i]
    proposed_new_bin=energy
    stop = False

    while not stop:
        proposed_new_bin +=cont_binning
        dist = abs(bin_edges_only_gamma[i+1]-(proposed_new_bin-cont_binning))
      
        if (proposed_new_bin<bin_edges_only_gamma[i+1]):
            bin_edges =insert_bin(proposed_new_bin,bin_edges)
        elif (proposed_new_bin>bin_edges_only_gamma[i+1] and dist<cont_binning/2):
            bin_edges=remove_bin(proposed_new_bin-cont_binning,bin_edges)
            stop=True
        else:           
            stop=True

print(bin_edges)


def bin_array_to_string(bin_array: np.ndarray) -> str:
    """
    Convert a NumPy array of bins into a string format representing bin sections.

    Parameters:
    - bin_array (np.ndarray): NumPy array of bin values.

    Returns:
    - str: A string in the format "low:bin:high" comma-separated, representing bin sections.
    """

    result = []
    current_start = bin_array[0]
    current_end = bin_array[0]
    current_diff = None

    for i in range(1, len(bin_array)):
        diff = bin_array[i] - bin_array[i - 1]

        if current_diff is None or diff == current_diff:
            current_diff = diff
            current_end = bin_array[i]
        else:
            result.append(f"{current_start}:{current_diff}:{bin_array[i - 1]}")
            current_start = bin_array[i-1]
            current_end = bin_array[i]
            current_diff = diff

    result.append(f"{current_start}:{current_diff}:{bin_array[-1]}")

    return ",".join(result)


print(bin_array_to_string(np.array([500,510,520,530,535,540,545,547,549,559,1000])))



### get the binning string
widths= np.diff(bin_edges)
print(bin_array_to_string(bin_edges))




def get_hist(obj):
    return obj.to_hist()



outfile="../hmixfit/results/hmixfit-l200a_vancouver_v0_5_rebin/histograms.root"


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.6,
}


with uproot.open(outfile) as f:

    hist_icpc = get_hist(f["l200a_vancouver23_dataset_v0_2_icpc"]["originals"]["fitted_data"])
    hist_icpc_rebin=utils.variable_rebin(hist_icpc,bin_edges)
    hist_icpc_rebin=utils.normalise_histo(hist_icpc_rebin)
    hist_icpc=hist_icpc[500:3000][hist.rebin(1)]
    
print(hist_icpc)    
hist_icpc=utils.normalise_histo(hist_icpc)





fig, axes_full = lps.subplots(1, 1, figsize=(4,3), sharex=True)
hist_icpc.plot(**style,ax=axes_full,color=vset.blue,histtype="fill",alpha=0.6,label="1 keV flat binning")
hist_icpc_rebin.plot(**style,ax=axes_full,color=vset.orange,label="variable binned")

axes_full.set_xlabel("Energy [keV]")
axes_full.set_ylabel("counts/keV")

axes_full.set_yscale("linear")
axes_full.legend(loc="upper right")
axes_full.set_xlim(565,3000)
plt.show()

