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

my_colors=[mset.indigo,vset.blue,vset.cyan,vset.teal,vset.red,vset.orange,vset.magenta,mset.rose,mset.wine,mset.purple]
plt.rc('axes', prop_cycle=plt.cycler('color',my_colors))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}



### first read the priors json

priors_file ="priors.json"
pdf_path = "../hmixfit/inputs/pdfs/l200a-pdfs_vancouver23/"
data_path="../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v0.1.root"
spec="mul_surv"
with open(priors_file,"r") as file_cfg:
    priors =json.load(file_cfg)

livetime=0.36455
print(json.dumps(priors,indent=1))

print(utils.priors2table(priors))
order=["fiber_shroud","sipm","birds_nest","wls_reflector","pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud"]



# extract a prior plot it and generate random numbers
rvs={}
quantiles={}
pdfs={}
pdfs_1={}
mc_list=[]
with PdfPages('plots/priors/prior_distributions.pdf') as pdf:
    for comp in priors["components"]:

    
        rv1,high1,range_ci = utils.extract_prior(comp["prior"])

        point_est = comp["best_fit"]
        low_err = point_est-range_ci[0]
        high_err = range_ci[1]-point_est
        quantiles[comp["name"]]=(point_est,low_err,high_err)
        samples1=np.array(rv1.rvs(size=100000))
        utils.plot_pdf(rv1,high1,samples1,pdf,name = comp["name"])
        rvs[comp["name"]]=rv1

        mc_pdf,N = utils.get_pdf_and_norm(pdf_path+comp["file"],b=10,r=(0,4000))
        mc_pdf_1,N = utils.get_pdf_and_norm(pdf_path+comp["file"],b=1,r=(0,4000))

        pdf_scale=utils.scale_hist(mc_pdf,1/N)
        pdf_scale_1=utils.scale_hist(mc_pdf_1,1/N)
        pdfs[comp["name"]]=pdf_scale
        pdfs_1[comp["name"]]=pdf_scale_1
        mc_list.append({"components":{comp["name"]:{}},"root-file":comp["file"]})

### now we need the PDFs themself (get it with uproot)


for type in ["Bi212Tl208","Pb214Bi214"]:
    labels=[]
    y=[]
    y_low=[]
    y_high=[]

    for comp in  quantiles.keys():
        if (type in comp):
            labels.append(comp)
            y.append(quantiles[comp][0])
            y_low.append(quantiles[comp][1])
            y_high.append(quantiles[comp][2])
    labels=np.array(labels)
    y=np.array(y)
    y_low=np.array(y_low)
    y_high=np.array(y_high)

    utils.make_error_bar_plot(np.arange(len(labels)),labels,y,y_low,y_high,type,save_path="plots/priors/prior_",show=False)

#### add possibility to get efficiencies

## first get a detector type map 
det_types,name = utils.get_det_types("sum")


### now a regions map
peaks=["Tl2615","Bi1764"]
energies=[2615,1764]
regions = {"fit_range":(565,4000)}

for peak,energy in zip(peaks,energies):
    regions[peak]=(energy-5,energy+5)
    




with PdfPages('plots/priors/exp_contributions.pdf') as pdf:
    index_Bi =0
    index_Tl=0

    
    for comp in priors["components"]:
        point_est = comp["best_fit"]
        upper_limit=0
        if (point_est==0):
            rv1,high,range_ci = utils.extract_prior(comp["prior"])
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


### get the total 
total = copy.deepcopy(total_Tl)
total+=total_Bi

output=utils.slow_convolve(priors,pdfs,rvs)
data = utils.get_data(data_path,b=10,r=(0,4000))
output*=livetime
pdf_tmp = utils.vals2hist(output[:,0],copy.deepcopy(total_Tl))
fig, axes = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})
utils.plot_mc_no_save(axes,pdf_tmp,name="",linewidth=0.6)


### now look at one bin
energies= pdf_tmp.axes.centers[0]
lows=[]
highs=[]
with PdfPages('plots/priors/bin_contents.pdf') as pdf:

    for bin in range(len(output[:,0])):

            fig, axes = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})
        
            low= np.percentile(output[bin,:],16)
            high=np.percentile(output[bin][:],90)
            lows.append(low)
            highs.append(high)
            if (abs(energies[bin]-2615)<5 or abs(energies[bin]-583)<5 or abs(energies[bin]-1764)<5 or abs(energies[bin]-2204)<5):

                axes.hist(output[bin,:],histtype="step",color=vset.red,bins=100,range=(0,max(output[bin,:])),label="Distribution")
                point_est = total[bin]
                axes.plot([point_est,point_est],[0,axes.get_ylim()[1]],color=vset.red,label="Point estimate")

                axes.axvspan(low,high, alpha=0.3, color='gray', label="68 pct. c.i. region")
                axes.set_title("Prediction for bin {}".format(energies[bin]))
                axes.set_ylabel("Prob [arb]")
                axes.set_xlabel("Counts")
                axes.legend(loc="best")
                pdf.savefig()
                point_est = (high+low)/2.
                if abs(energies[bin]-2615)<5:
                    pred_2615 = point_est/livetime
                    error_low_2615 = (point_est-low)/livetime
                    error_high_2615 =(high-point_est)/livetime
                if abs(energies[bin]-1764)<5:
                    pred_1764 = point_est/livetime
                    error_low_1764 = (point_est-low)/livetime
                    error_high_1764 =(high-point_est)/livetime




#### save a plot with the expected counts
mc_list= np.array(mc_list)
## now get the efficiency 
effs  = utils.get_total_efficiency(det_types,None,"mul_surv",regions,pdf_path,mc_list=mc_list)

### get counts in gamma lines (data)
data_file = uproot.open(data_path)
counts_2615_data,low_2615_data,high_2615_data =utils.get_peak_counts(2615,"Tl208",data_file,livetime)
counts_1764_data,low_1764_data,high_1764_data =utils.get_peak_counts(1764,"Bi214",data_file,livetime)

print("We see in 2615  keV {} - {} + {} counts".format(counts_2615_data,low_2615_data,high_2615_data))
print("We see in 1764  keV {} - {} + {} counts".format(counts_1764_data,low_1764_data,high_1764_data))


with PdfPages('plots/priors/prior_values.pdf') as pdf:

    for type in ["Bi212Tl208","Pb214Bi214"]:
        
        for region in effs["all"].keys():

            if (region=="Tl2615" and type=="Pb214Bi214"):
                continue
            if (region=="Bi1764" and type=="Bi212Tl208"):
                continue

            labels=[]
            categories=[]
            y=[]
            y_low=[]
            y_high=[]

            data_band=None
            for comp in  priors["components"]:
                if (type in comp["name"]):
                    labels.append(comp["name"])
                    eff = effs["all"][region][comp["name"]]

                    y.append(eff*quantiles[comp["name"]][0])
                    y_low.append(eff*quantiles[comp["name"]][1])
                    y_high.append(eff*quantiles[comp["name"]][2])
                    categories.append(comp["type"])


            if region=="Tl2615":
                labels.append("Total")
                y.append(pred_2615)
                y_low.append(error_low_2615)
                y_high.append(error_high_2615)
                categories.append("total")
                data_band =(counts_2615_data,low_2615_data,high_2615_data)

            if region=="Bi1764":
                labels.append("Total")
                y.append(pred_1764)
                y_low.append(error_low_1764)
                y_high.append(error_high_1764)
                data_band =(counts_1764_data,low_1764_data,high_1764_data)
                categories.append("total")

            labels=np.array(labels)
            y=np.array(y)
            y_low=np.array(y_low)
            y_high=np.array(y_high)
            categories=np.array(categories)
            
            utils.make_error_bar_plot(np.arange(len(labels)),labels,y,y_low,y_high,region,save_path="plots/priors/prior_",show=False,obj="fit_range",pdf=pdf,
                                      data_band=data_band,categories=categories)





lows=np.array(lows)
highs=np.array(highs)
pdf_high = utils.vals2hist(highs,copy.deepcopy(total_Tl))

### lets make a plot comparing the different shapes
with PdfPages('plots/priors/shapes.pdf') as pdf:
   
    utils.compare_mc(239j,pdfs,"Bi212Tl208",order,2615j,pdf,0,4000,"log")
    utils.compare_mc(583j,pdfs,"Bi212Tl208",order,2615j,pdf,500,700,"linear",linewidth=.8)
    utils.compare_mc(727j,pdfs,"Bi212Tl208",order,2615j,pdf,700,900,"linear",linewidth=.8)
    utils.compare_mc(2104j,pdfs,"Bi212Tl208",order,2615j,pdf,1800,2700,"linear")
    utils.compare_mc(100j,pdfs,"Pb214Bi214",order,1764j,pdf,0,4000,"log")
    utils.compare_mc(609j,pdfs,"Pb214Bi214",order,1764j,pdf,500,1000,"linear",linewidth=.8)
    utils.compare_mc(2204j,pdfs,"Pb214Bi214",order,1764j,pdf,1900,2500,"linear",linewidth=.8)

### also get the data

with PdfPages('plots/priors/total_contributions.pdf') as pdf:
        utils.plot_mc(total_Tl,"Total $^{212}$Bi+ $^{208}$Tl",pdf,data=data)
        utils.plot_mc(total_Bi,"Total $^{214}$Pb+ $^{214}$Bi",pdf,data=data)
        utils.plot_mc(total,"Total",pdf,data=data,pdf2=pdf_high)
        utils.plot_mc(total,"Total",pdf,data=data,range_x=(1900,2700),range_y=(0.01,500),pdf2=pdf_high)

