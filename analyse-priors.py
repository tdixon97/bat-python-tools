"""
analyse-priors.py
Author: Toby Dixon
Date : 30th November 2023

This is a script to analyse the screening measurements for LEGEND-200, estimate the total bkg (from screening) etc.

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
from legendmeta import LegendMetadata
from matplotlib.backends.backend_pdf import PdfPages

vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}



### first read the priors json

priors_file ="priors.json"

with open(priors_file,"r") as file_cfg:
    priors =json.load(file_cfg)


print(json.dumps(priors,indent=1))

# extract a prior plot it and generate random numbers
with PdfPages('plots/priors/prior_distributions.pdf') as pdf:
    for comp in priors["components"]:

    
        rv1,high1 = utils.extract_prior(comp["prior"])
        samples1=np.array(rv1.rvs(size=100000))
        utils.plot_pdf(rv1,high1,samples1,pdf,name = comp["name"])

