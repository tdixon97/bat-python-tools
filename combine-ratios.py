""" combine-ratios.py
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk)
This code performs a combination of measurements using BAT, modelling them as a sum of MC predictions.
"""

#import juliacall
#from batty import BAT_sampler, BAT, Distributions, jl
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon

from collections import OrderedDict
import uproot
#import juliacall
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
from hist import Hist
import hist
import os
import sys
import argparse
import re
import utils
import variable_binning
import json
from copy import copy
from copy import deepcopy
from matplotlib.backends.backend_pdf import PdfPages
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
lset = tc.tol_cset('light')

cmap=tc.tol_cmap('iridescent')
cmap.set_under('w',1)
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))



def get_directories(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


style = {

    "yerr": False,
    "flow": None,
    "lw": 0.7,
}
def dir_convert(input):
    if ("M2_E1" in input):
        return "mul2_e1"
    elif ("M2_E2" in input):
        return "mul2_e2"
    elif ("M1" in input):
        return "mul_surv"
    else:
        raise ValueError("The directory doesnt correspond to a category for the fit")
### first get the data
### --------------------------------

data_path = "gamma_line_output/histogram/per_toby/"
save_mc_ratio="out/mc_ratios.json"

directories = get_directories(data_path)

hist_map={}
for dir in directories:
    hist_map[dir_convert(dir)]={}
    print(dir)
    for file in os.listdir(data_path+dir):
        with uproot.open(data_path+dir+"/"+file) as data_file:
            hists= utils.get_list_of_not_directories(file  = data_file)

            hists_sel = [hist for hist in hists if "h1" in hist and "E0_height" in hist]
            if (len(hists_sel)>1):
                raise ValueError(f"Error multiple suitable hists for found  in {file}")
            elif (len(hists_sel)==0):
                raise ValueError(f"No suitable hists for found  in {file}")

            hists_sel=hists_sel[0].split(";")[0]
            h=data_file[hists_sel].to_hist()
            for i in range(h.size-2):
                if (h[i]>0):
                    h[i]=np.log(h[i])
                else:
                    h[i]=-1E2
            if (file[0:-5]!="3198"):
                hist_map[dir_convert(dir)][file[0:-5]]=h
            else:
                hist_map[dir_convert(dir)]["3197"]=h

### now get the MC
####------------------

with open(save_mc_ratio, "r") as json_file:
    ratios = json.load(json_file)


print(hist_map)
#print(json.dumps(ratios,indent=1))

comp_list=["pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud","fiber_shroud","sipm","birds_nest","wls_reflector"]
spec_ref="mul_surv"
peak_ref="2615"
type_decay="Bi212Tl208"

peaks=[("mul_surv","583"),("mul_surv","3197"),("mul_surv","3126"),("mul2_e1","583"),("mul2_e2","583"),("mul2_e1","2615")]
w,x= hist_map[spec_ref][peak_ref].to_numpy()
n_ref = x[np.argmax(w)]
print(n_ref)
priors=[]

for peak in peaks:
    priors.append(Distributions.Uniform(0,1.2))
print(priors)
def likelihood_slow(x,peaks,comp_list,ratios,type_decay,n_ref):
    """
    The likelihood function is just a sum over evaluations of the posteriors
    """

    for peak in peaks:

        for comp in comp_list:
            N = x.fracs[i]*ratios[type_decay+"_"+comp][peak[0]][peak[1]]*n_ref


