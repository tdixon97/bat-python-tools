""" calibration-comparison.py
Python script to plot the activites from the hmixfit fit.
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk)
"""

from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
from datetime import datetime, timezone

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
import os
import numpy as np
from IPython.display import display
import pandas as pd
from hist import Hist
from matplotlib.backends.backend_pdf import PdfPages
lps.use('legend')
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))

def time_to_unix(str):
    datetime_obj = datetime.strptime(str, "%Y%m%dT%H%M%SZ")
    unix_time = int(datetime_obj.timestamp())

    return unix_time

def data_compare(spec,chan,pos):

    path = "/home/tdixon/LEGEND/BackgroundModel/hmixfit/inputs/data/datasets/calibration"

    files = os.listdir(path)
    counts=[]
    names=[]
    tstarts=[]
    deltat=[]
    bins=[]
    t0=None
    for file in sorted(files):
        full_path =os.path.join(path,file)
        if (pos in full_path):
            period=file.split(".")[-2].split("-")[-3]
            run=file.split(".")[-2].split("-")[-2]

            with uproot.open(full_path) as f:

                h=f["hit"][chan].to_hist()
                counts.append(utils.integrate_hist(h,2610,2620))
                names.append(period+"-"+run)

            meta_data = f"cfg/DataKeys/cal/{period}/l200-{period}-{run}-cal-T%-keys.json"
            with open(meta_data,"r") as json_file:
                cfg_dict =json.load(json_file)
            tstart = cfg_dict["info"]["source_positions"][pos]["keys"][0].split("-")[-1]
            livetime =int(cfg_dict["info"]["source_positions"][pos]["livetime_in_s"])
            if (livetime==0):
                continue
            tstarts.append(tstart)
            deltat.append(livetime)
            if (t0 is None):
                t0=time_to_unix(tstart)/60/60
            bins.append(time_to_unix(tstart)/60/60-t0)
            bins.append((time_to_unix(tstart)+livetime)/60/60-t0)

    # get the metadata
    histo_time =( Hist.new.Variable(bins).Double())
    for ts,dt,count in zip(tstarts,deltat,counts):

        histo_time[((time_to_unix(ts)+dt/2.)/60/60-t0)*1j]=count

    widths= np.diff(histo_time.axes.edges[0])
    centers=histo_time.axes.edges[0]

    x=[]
    y=[]
    ey_low=[]
    ey_high=[]
    fig, axes_full = lps.subplots(1, 1, figsize=(4,3), sharex=True)

    for i in range(histo_time.size-2):
        if (histo_time[i]>0 and widths[i]>1):
        
    
            norm=(widths[i])
            x.append(centers[i])
            y.append(histo_time[i]/norm)
            el,eh =utils.get_error_bar(histo_time[i])
            ey_low.append(el/norm)  
            ey_high.append(eh/norm)
    xrange = np.linspace(0,np.max(x),100000)
    
    axes_full.errorbar(x=x,y=y,yerr=[np.abs(ey_low),np.abs(ey_high)],color=vset.blue,fmt="o",ecolor="grey",label="With OB")
    axes_full.plot(xrange,y[0]*np.exp(-np.log(2)*np.array(xrange)/(1.92*365*24)))
    axes_full.set_ylim(0,100000)
    axes_full.set_xlabel("Time [hours]")
    axes_full.set_ylabel("2615 keV counts/hour")
    axes_full.set_title(f"{chan} {pos}")
    plt.show()
data_compare("hit","ch1078400","pos1")
data_compare("hit","ch1078400","pos2")

data_compare("hit","ch1078405","pos1")
data_compare("hit","ch1078405","pos2")


data_compare("hit","ch1084800","pos1")
data_compare("hit","ch1084800","pos2")


data_compare("hit","ch1084803","pos1")
data_compare("hit","ch1084803","pos2")
