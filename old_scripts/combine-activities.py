"""
A script to combine gaussian / exponentials
Author: toby dixon (toby.dixon.23@ucl.ac.uk)
"""

from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')

from collections import OrderedDict
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
import hist
import argparse
import re
import utils
import json
import time
import warnings
import sys
from legendmeta import LegendMetadata
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import truncnorm
import os
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}
def sample_truncnorm(mean,std):
    lower=0
    upper=1000

    a = (lower - mean) / std
    b = (upper - mean) / std

    samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=int(1e6))

    return samples

def sample_norm(mean,std):
    lower=-1000
    upper=1000

    a = (lower - mean) / std
    b = (upper - mean) / std

    samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=int(1e6))

    return samples

def plot_gaus(mean,std,axes,label):
    x = np.linspace(0,1000, 1000)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std**2))

    axes.plot(x,2*y,label=label)
measures=np.array([0,0,0,0,0,0,0,0])
errors = np.array([50,50,50,50,50,50,50,50])
samples=np.zeros(int(1e6))
samples_norm=np.zeros(int(1e6))

for meas,err in zip(measures,errors):
    samples+=sample_truncnorm(meas,err)
    samples_norm+=sample_norm(meas,err)

fig, axes = lps.subplots(1, 1,figsize=(4, 4), sharex=True, gridspec_kw = { "hspace":0})

axes.hist(samples,bins=300,range=(0,1000),alpha=0.4,lw=0.5,color=vset.blue,density=True,label="samples from trunc. gaus")
axes.hist(samples_norm,bins=300,range=(0,1000),alpha=0.4,lw=0.5,color=vset.red,density=True,label="samples from gaus")
plot_gaus(np.sum(meas),np.sqrt(np.sum(np.power(errors,2))),axes,"analytic prediction")
axes.set_xlabel("Activity [$\mu$Bq]")
axes.set_ylabel("Probability [arb]")
plt.legend(loc="best")
plt.show()
