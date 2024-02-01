#!/usr/bin/env python
# coding: utf-8

# ## Variable binning
# 
# Notebook to make some tests of a variable binning scheme for the background model fit.
# 
# Author: Toby Dixon (toby.dixon.23@ucl.ac.uk)

import re
import pandas as pd
from iminuit import Minuit, cost

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
    """
    Remove a elements with value bin from the bins array while ensuring that the element at index 0 is never removed.

    Parameters:
    - bin (int): The value of the bin to be removed.
    - bins (numpy.ndarray): The array of bins.

    Returns:
    numpy.ndarray: A new array of bins with the specified bin removed,
                   or the original array if bin is equal to the value at index 0.
    """
     
    ## Never remove the first bin
    if bin == bins[0]:
        return bins

    return bins[bins!=bin]



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


def compute_binning(gamma_energies=np.array([583,609,911,1460,1525,1764,2204,2615]),low_energy=565,high_energy=4000,gamma_binning_high=10,gamma_binning_low=6,cont_binning=15):
    """
    Function to compute a variable binning given some inputs
    """
    bin_edges = np.array([low_energy,high_energy])

    for i in range(len(gamma_energies)):

        ### check first for overlaps
        energy = gamma_energies[i]
        if (energy<1000):
            gamma_binning=gamma_binning_low
        else:
            gamma_binning=gamma_binning_high

        if (i!=len(gamma_energies)-1):

            if (energy+gamma_binning/2>gamma_energies[i+1]-gamma_binning/2):
                raise ValueError("Error: Gamma lines are too close - revise")
            
        bin_edges=insert_bin(int(energy-gamma_binning/2),bin_edges)
        bin_edges=insert_bin(int(energy+gamma_binning/2),bin_edges)


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
            
            ## never remove a gamma line bin
            is_gamma = np.isin(proposed_new_bin-cont_binning,bin_edges_only_gamma)
            if (proposed_new_bin<bin_edges_only_gamma[i+1] ):
                bin_edges =insert_bin(proposed_new_bin,bin_edges)
            elif (proposed_new_bin>bin_edges_only_gamma[i+1] and dist<cont_binning/2 and not is_gamma):
                bin_edges=remove_bin(proposed_new_bin-cont_binning,bin_edges)
                stop=True
            else:           
                stop=True
          
    return bin_edges

def remove_duplicates(bin_edges):
    return np.unique(bin_edges)

def reso_curve(x,a,b):
    return np.sqrt(a+x*b)

def M2_reso_analysis(json_path:str="gamma_line_output/M2_resolutions_p3_p7_20240131.json",
                     json_path_M1:str="gamma_line_output/M1_resolutions_p3_p7_20240131.json"
                     ):
    """Analysis of the M2 resolutions for M2 events"""


    with open(json_path, 'r') as json_file:
        M2_reso =json.load(json_file)


    with open(json_path_M1, 'r') as json_file:
        M1_reso =json.load(json_file)

    specs=["M2_E1_raw","M2_E2_raw"]
    fig, axes= lps.subplots(1, 1,figsize=(3, 4), sharex=True, gridspec_kw = { "hspace":0})
    peaks=["583","609","1764","2615"]
    cs=[vset.teal,vset.blue,vset.orange]
    x=[]
    y=[]
    el=[]
    eh=[]
    for idx,spec in enumerate(specs):
       
        for key in M2_reso[spec]:
            if (key not in peaks):
                continue
            
            x.append(float(key))
            y.append(M2_reso[spec][key]["FWHM"])
            eh.append(M2_reso[spec][key]["mode_to_q84"])
            el.append(M2_reso[spec][key]["mode_to_q16"])
        if (idx==0):
            continue
        axes.errorbar(np.array(x)+2*idx,y,yerr=[el,eh],color=cs[idx],fmt="o",label="M2 data",linewidth=0.5,markersize=2)
        axes.set_xlabel("Energy [keV]")
        axes.set_ylabel("Reso FWHM [keV]")
        xi=np.linspace(3000,0,100)
        a=M1_reso["All"]["a"]
        b=M1_reso["All"]["b"]
        axes.plot(xi,np.sqrt(a+b*xi),color=vset.red,linestyle="--",label="M1 resolution curve")
        axes.set_xlim(0,3000)
        axes.set_ylim(0,5)

    x=np.array(x)
    y=np.array(y)
    eh=np.array(eh)
    el=np.array(el)
    ### make the fit
    likelihood=utils.create_graph_likelihood(reso_curve,x,y,el,eh)
    guess=(1,0.001)
    m = Minuit(likelihood, *guess)
    m.limits[0]=(0,5)
    m.limits[1]=(0,0.1)
    m.migrad()
    axes.plot(xi,np.sqrt(m.values[0]+m.values[1]*xi),color=vset.teal,linestyle="--",label=f"Fit to M2")
    print(m)
    m.minos()
    print(m)
    axes.legend()

    plt.show()

def main():


    ## at some point read these files
    
    gammas = np.array([511, 583, 609, 727, 835, 911, 969, 1120, 1173, 1378, 1461, 1525, 1588, 1730, 1765, 2104, 2119, 2204, 2448, 2615])
    bin_edges=compute_binning(gamma_energies=gammas,low_energy=500,high_energy=4000,gamma_binning_low=6,gamma_binning_high=10,cont_binning=20)
    bin_edges_coax=compute_binning(gamma_energies=gammas,low_energy=500,high_energy=4000,gamma_binning_low=15,gamma_binning_high=15,cont_binning=20)

    gammas_M2 = np.array([277,511,583,609,768,911,934,1120,1173,1238,1333,1408,1525,2104,2615])
    bin_edges_M2=compute_binning(gamma_energies=gammas_M2,low_energy=200,high_energy=3000,gamma_binning_low=20,gamma_binning_high=20,cont_binning=20)
    bin_edges_M2_2D=compute_binning(gamma_energies=gammas_M2,low_energy=200,high_energy=3000,gamma_binning_low=20,gamma_binning_high=20,cont_binning=50)

    bin_edges=np.unique(bin_edges)
    bin_edges_coax=np.unique(bin_edges_coax)
    
    ### get the binning string

    widths= np.diff(bin_edges)
    print("bins ",bin_edges)
    print("bins coax ",bin_edges_coax)
    print("bins m2 ",bin_edges_M2)
    print("bins 2D ",bin_edges_M2_2D)
    print("bins ",bin_array_to_string(bin_edges))
    print("bins coax ",bin_array_to_string(bin_edges_coax))
    print("bins m2 ",bin_array_to_string(bin_edges_M2))
    print("bins 2d ",bin_array_to_string(bin_edges_M2_2D))




    def get_hist(obj):
        return obj.to_hist()



    outfile="../hmixfit/results/hmixfit-l200a_vancouver_v0_5_rebin/histograms.root"


    style = {
        "yerr": False,
        "flow": None,
        "lw": 0.6,
    }

    for dtype in ["ppc","icpc","bege","coax"]:
        with uproot.open(outfile) as f:

            hist_icpc = get_hist(f["l200a_vancouver23_dataset_v0_2_{}".format(dtype)]["originals"]["fitted_data"])
            if (dtype!="coax"):
                hist_icpc_rebin=utils.variable_rebin(hist_icpc,bin_edges)
            else:
                hist_icpc_rebin=utils.variable_rebin(hist_icpc,bin_edges_coax)

            hist_icpc_rebin=utils.normalise_histo(hist_icpc_rebin)
            hist_icpc=hist_icpc[500:3000][hist.rebin(1)]
            
        hist_icpc=utils.normalise_histo(hist_icpc)





        fig, axes_full = lps.subplots(1, 1, figsize=(3,3), sharex=True)
        hist_icpc.plot(**style,ax=axes_full,color=vset.blue,histtype="fill",alpha=0.6,label="1 keV flat binning")
        hist_icpc_rebin.plot(**style,ax=axes_full,color=vset.orange,label="variable binned")
        axes_full.set_xlabel("Energy [keV]")
        axes_full.set_ylabel("counts/keV")
        axes_full.set_yscale("log")
        axes_full.legend(loc="upper right")
        axes_full.set_xlim(500,4000)
        plt.savefig("plots/binning/check_{}.pdf".format(dtype))

        for E in [600,1500,2615]:
            axes_full.set_xlabel("Energy [keV]")
            axes_full.set_ylabel("counts/keV")

            axes_full.set_yscale("linear")
            axes_full.legend(loc="upper right")
            axes_full.set_xlim(E-50,E+50)
            
            axes_full.set_ylim(0,1.1*np.max(hist_icpc[(E-50)*1j:(E+50)*1j].values()))
            plt.savefig("plots/binning/check_{}_{}.pdf".format(dtype,E))
        #plt.show()

    pass
if __name__ == "__main__":
    #main()
    M2_reso_analysis()