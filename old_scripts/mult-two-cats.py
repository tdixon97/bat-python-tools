"""
mult-two-cats.py
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk)

"""
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
import sys
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
from hist import Hist
import time
import warnings
from legendmeta import LegendMetadata
from matplotlib.backends.backend_pdf import PdfPages


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.6,
}

vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
lset = tc.tol_cset('light')

cmap=tc.tol_cmap('iridescent')
cmap.set_under('w',1)
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))
 
def plot_2d_hist(hist,axes):
    w, x, y = hist.to_numpy()

    my_cmap = mpl.colormaps.get_cmap('BuPu')
    my_cmap.set_under('w')
    mesh = axes.pcolormesh(x, y, w.T, cmap=my_cmap,vmin=0.5)
    cbar =fig.colorbar(mesh,ax=axes)
    cbar.set_label("Counts")

def get_events(m2_histos,events):
    
    ### build the histo
    hist_out =( Hist.new.Reg(7,0,7).Reg(7, 0, 7).Double())

    if events["var"]=="E1":
        axis=1
    elif (events["var"]=="E2"):
        axis=0
    else:
        axis=2
    for s in range(7):
        for f in range(7):

            if f"sd_{s}_fd_{f}" not in m2_histos:
                continue
            hist_tmp = m2_histos[f"sd_{s}_fd_{f}"]
            
            if (axis==2):
                hist_tmp =utils.project_sum(hist_tmp)
            
            else:
                hist_tmp =hist_tmp.project(axis)
            
            var =utils.integrate_hist(hist_tmp,events["Energy"]-3,events["Energy"]+3)
            
            hist_out[s,f]+=var
           
    w=hist_out.values()
    
    return hist_out


def get_events_1d(m2_histos,events):
    
    ### build the histo
    hist_out =( Hist.new.Reg(7,0,7).Double())

    if events["var"]=="E1":
        axis=1
    elif (events["var"]=="E2"):
        axis=0
    else:
        axis=2
    for s in range(7):
        for f in range(7):

            if f"sd_{s}_fd_{f}" not in m2_histos:
                continue
            hist_tmp = m2_histos[f"sd_{s}_fd_{f}"]
            
            if (axis==2):
                hist_tmp =utils.project_sum(hist_tmp)
            
            else:
                hist_tmp =hist_tmp.project(axis)
            
            var =utils.integrate_hist(hist_tmp,events["Energy"]-3,events["Energy"]+3)
            
            hist_out[s]+=var
           
    w=hist_out.values()
    
    return hist_out      

def make_plot(m2_histos,axis,type_cat,low=400,high=700):
    
    ### build the histo
    if (axis==0):
        hist_out =( Hist.new.Reg(3000,0,3000).Reg(10, 0, 10).Double())
    elif (axis==1):
        hist_out =( Hist.new.Reg(3000,0,3000).Reg(10, 0, 10).Double())
    elif (axis==2):
        hist_out =( Hist.new.Reg(6000,0,6000).Reg(10, 0, 10).Double())

    for s in range(10):
        for f in range(10):

            if f"sd_{s}_fd_{f}" not in m2_histos:
                continue
            hist_tmp = m2_histos[f"sd_{s}_fd_{f}"]
            
            if (axis==2):
                hist_tmp =utils.project_sum(hist_tmp)
               
            else:
                hist_tmp =hist_tmp.project(axis)
            
            if (type_cat=="s"):
                for i in range(hist_tmp.size-2):
                    hist_out[i,s]+=hist_tmp[i]
            if (type_cat=="f"):
                for i in range(hist_tmp.size-2):
                    hist_out[i,f]+=hist_tmp[i]
    
    hist_out=hist_out[low:high,0:7][hist.rebin(10),hist.rebin(1)]
    return hist_out
                                                                
data_path="../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v0.4.root"

m2_histos={}
string_diff=np.arange(7)
floor_diff=np.arange(8)
names_m2 =[f"sd_{item1}_fd_{item2}" for item1 in string_diff for item2 in floor_diff]
names_m2.extend(["all","cat_1","cat_2","cat_3"])

with uproot.open(data_path,object_cache=None) as f:
    for name in names_m2:
        m2_histos[name]=f["mul2_surv_2d"][name].to_hist()[0:3000,0:3000]

### plot the Delta S vs E
with PdfPages("plots/mult_two/new_cats.pdf") as pdf:

    h=make_plot(m2_histos,0,"s",400,700)
    fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
    axes.set_xlabel("Energy 2 [keV]")
    axes.set_ylabel("$\Delta$ string")
    plot_2d_hist(h,axes)
    pdf.savefig()

    h=make_plot(m2_histos,1,"s",500,3000)
    fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
    axes.set_xlabel("Energy 1 [keV]")
    axes.set_ylabel("$\Delta$ string")
    plot_2d_hist(h,axes)
    pdf.savefig()


    h=make_plot(m2_histos,2,"s",500,3000)
    fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
    axes.set_xlabel("Energy Sum [keV]")
    axes.set_ylabel("$\Delta$ string")
    plot_2d_hist(h,axes)
    pdf.savefig()

    h=make_plot(m2_histos,0,"f",400,700)
    fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
    axes.set_xlabel("Energy 2 [keV]")
    axes.set_ylabel("$\Delta$ position")
    plot_2d_hist(h,axes)
    pdf.savefig()

    h=make_plot(m2_histos,1,"f",500,3000)
    fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
    axes.set_xlabel("Energy 1 [keV]")
    axes.set_ylabel("$\Delta$ positon")
    plot_2d_hist(h,axes)
    pdf.savefig()

    h=make_plot(m2_histos,2,"f",500,3000)
    fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
    axes.set_xlabel("Energy Sum [keV]")
    axes.set_ylabel("$\Delta$ position")
    plot_2d_hist(h,axes)
    pdf.savefig()




    sigs=[
        
        {
            "name":"583 keV M2-E1",
            "var":"E1",
            "Energy":583
            },

        {
            "name":"583 keV M2-E2",
            "var":"E2",
            "Energy":583
            },
        {
            "name":"609 keV M2-E1",
            "var":"E1",
            "Energy":609
            },
        {
            "name":"609 keV M2-E2",
            "var":"E2",
            "Energy":609
            },
        {
            "name":"2615 keV M2-E1",
            "var":"E1",
            "Energy":2615
            },
        {
            "name":"1525 keV M2-E1",
            "var":"E1",
            "Energy":1525
            },
        {
            "name":"1525 keV M2-S",
            "var":"S",
            "Energy":1525
            },
        {
            "name":"1461 keV M2-S",
            "var":"S",
            "Energy":1461
            }
        

    ]
    for sig in sigs:
        h=get_events(m2_histos,sig)
        print(h)
        fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
        axes.set_xlabel("$\Delta$ string")
        axes.set_ylabel("$\Delta$ position")
        axes.set_title(sig["name"])
        plot_2d_hist(h,axes)
        pdf.savefig()

    for sig in sigs:
        h=get_events_1d(m2_histos,sig)
        fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
        axes.set_xlabel("$\Delta$ string")
        axes.set_ylabel("Counts")
        axes.set_title(sig["name"])
        h.plot(ax=axes,**style,color=vset.blue,histtype="fill",alpha=0.4)
        axes.set_xlim(0,6)
        pdf.savefig()






