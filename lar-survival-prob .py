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
