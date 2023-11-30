"""
analyse-priors.py
Author: Toby Dixon
Date : 30th November 2023

This is a script to analyse the screening measurements for LEGEND-200, estimate the total bkg (from screening) etc.

"""


from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')

import copy
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
sunset=tc.tol_cmap('sunset')
sunset=  ['#332288','#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF',
                '#EAECCC',  '#FDB366', '#F67E4B', '#DD3D2D',
                vset.red]

my_colors=[vset.red,vset.orange,vset.magenta,mset.rose,mset.wine,mset.purple,mset.indigo,vset.blue,vset.cyan,vset.teal]
plt.rc('axes', prop_cycle=plt.cycler('color',my_colors))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}



### first read the priors json

priors_file ="priors.json"
pdf_path = "../hmixfit/inputs/pdfs/l200a-pdfs_vancouver23/"
spec="mul_surv"
with open(priors_file,"r") as file_cfg:
    priors =json.load(file_cfg)

livetime=0.36455
print(json.dumps(priors,indent=1))
order=["pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud","fiber_shroud","sipm","birds_nest","wls_reflector"]
# extract a prior plot it and generate random numbers
rvs={}
pdfs={}
with PdfPages('plots/priors/prior_distributions.pdf') as pdf:
    for comp in priors["components"]:

    
        rv1,high1 = utils.extract_prior(comp["prior"])
        point_est = comp["best_fit"]
        samples1=np.array(rv1.rvs(size=100000))
        utils.plot_pdf(rv1,high1,samples1,pdf,name = comp["name"])
        rvs[comp["name"]]=rv1

        mc_pdf,N = utils.get_pdf_and_norm(pdf_path+comp["file"],b=10,r=(0,4000))

        pdf_scale=utils.scale_hist(mc_pdf,1/N)
     
        pdfs[comp["name"]]=pdf_scale

### now we need the PDFs themself (get it with uproot)

with PdfPages('plots/priors/exp_contributions.pdf') as pdf:
    index_Bi =0
    index_Tl=0
    for comp in priors["components"]:
        point_est = comp["best_fit"]
        upper_limit=0
        if (point_est==0):
            rv1,high = utils.extract_prior(comp["prior"])
            point_est= high
            upper_limit=1

        pdf_norm = utils.scale_hist(pdfs[comp["name"]],livetime*point_est)
        utils.plot_mc(pdf_norm,comp["name"],pdf)
        
        
        if (index_Tl==0 and upper_limit==0 and "Tl208" in comp["name"]):
            total_Tl = copy.deepcopy(pdf_norm)
            index_Tl+=1
        elif (upper_limit==0 and "Tl208" in comp["name"]):
            total_Tl+=pdf_norm
            index_Tl+=1
        
        if (index_Bi==0 and upper_limit==0 and "Bi214" in comp["name"]):
            total_Bi = copy.deepcopy(pdf_norm)
            index_Bi+=1
        elif (upper_limit==0 and "Bi214" in comp["name"]):
            total_Bi+=pdf_norm
            index_Bi+=1

### lets make a plot comparing the different shapes
with PdfPages('plots/priors/shapes.pdf') as pdf:
    index_Bi =0
    index_Tl=0
    fig, axes = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})
    max_583=0
    max_2104 = 0
    for name in order:
 
       
        Tl_counts =pdfs["Bi212Tl208_"+name][2615j]

        pdf_norm = utils.scale_hist(pdfs["Bi212Tl208_"+name],1/Tl_counts)

        if (pdf_norm[583j]>max_583):
            max_583 = pdf_norm[583j]
        if (pdf_norm[2104j]>max_2104):
            max_2104 =pdf_norm[2104j]
        axes=utils.plot_mc_no_save(axes,pdf_norm,name=name)
        
    axes.legend(loc='best',edgecolor="black",frameon=True, facecolor='white',framealpha=1,fontsize=8)    
    pdf.savefig()
    
    axes.set_yscale("linear")
    pdf.savefig()
    
    axes.set_ylim(0,1.2*max_583)
    axes.set_xlim(500,600)
    pdf.savefig()
    
    axes.set_ylim(0,0.5*max_583)
    axes.set_xlim(700,1000)
    pdf.savefig()
    
    axes.set_ylim(0,1.2*max_2104)
    axes.set_xlim(2000,2700)
    pdf.savefig()
### also get the data

with PdfPages('plots/priors/total_contributions.pdf') as pdf:
        utils.plot_mc(total_Tl,"Total $^{212}$Bi+ $^{208}$Tl",pdf)
        utils.plot_mc(total_Bi,"Total $^{214}$Pb+ $^{214}$Bi",pdf)

