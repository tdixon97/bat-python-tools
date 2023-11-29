"""
lar-survival-prob.py
Author: Toby Dixon
Date  : 26/11/2023
This script is meant to study the survival probability in different regions for the LAR cut.
Both in the data, and different MC contributions. 
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


vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}

peak = 2615
data_path = "../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v0.1.root"
file = uproot.open(data_path)

regions = {"Tl_peak":(2610,2620)}
det_types,name = utils.get_det_types("sum")
print(json.dumps(det_types,indent=1))
print(name)
data_counts = utils.get_data_counts("mul_surv",det_types,regions,file)
data_counts_lar = utils.get_data_counts("mul_lar_surv",det_types,regions,file)

print(json.dumps(data_counts,indent=1))