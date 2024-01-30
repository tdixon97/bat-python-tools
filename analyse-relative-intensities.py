"""
analyse-relative-intensities.py
Author: Toby Dixon
Date  : 1/12/2023
This script is meant to study relative intensitiies in gamma lines
"""

from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
import pandas
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
mset =tc.tol_cset('muted')

my_colors=[mset.indigo,vset.blue,vset.cyan,vset.teal,vset.red,vset.orange,vset.magenta,mset.rose,mset.wine,mset.purple]
plt.rc('axes', prop_cycle=plt.cycler('color',my_colors))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}



### first read the priors json

pdf_path = "../hmixfit/inputs/pdfs/l200a-pdfs_vancouver23/"
data_path="../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v0.2.root"
specs=["mul_surv","mul2_surv"]
priors_file ="priors.json"

with open(priors_file,"r") as file_cfg:
    priors =json.load(file_cfg)

livetime=0.36455
order=["fiber_shroud","sipm","birds_nest","wls_reflector","pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud"]
order2=["pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud","fiber_shroud","sipm","birds_nest","wls_reflector"]

### we need to extract the PDFs and get the counts in relevant gamma lines first

#type="Bi212Tl208"
type="Pb214Bi214"
if (type=="Bi212Tl208"):
    peaks=[583,727,861,1593,2104,2615,3197]
    main=2615
elif(type=="Pb214Bi214"):
    peaks=[609,1238,1378,1764,2017,2204,2447]
    main=2204



output_maps ={}
with PdfPages('plots/relative_intensity/pdfs_{}.pdf'.format(type)) as pdf:
    for spec in specs:
        output_map={}
        for comp in priors["components"]:
        
            output_tmp ={}
            if (type not in comp["name"]):
                continue

            for peak in peaks:

                pdf_tmp,N = utils.get_pdf_and_norm(pdf_path+comp["file"],b=10,r=(peak-15,peak+15),spectrum=spec)
                center= pdf_tmp[1]
                bkg = (pdf_tmp[2]+pdf_tmp[0])/2.
                value = center-bkg
                if (value<0):
                    value=0
                output_tmp[peak]=value
                utils.plot_mc(pdf_tmp,"Peak of {} in {}".format(peak,comp["name"]),pdf,range_x=(peak-15,peak+15),range_y=(1,center*2),scale="linear")

            output_map[comp["name"][11:]]=output_tmp
        output_maps[spec]=output_map


        ## normalise
        for comp in output_map.keys():
            norm  = output_maps["mul_surv"][comp][main]
            for peak in output_maps[spec][comp].keys():
                if ((peak==main and spec=="mul_surv")):
                    continue
                output_maps[spec][comp][peak]/=norm




## ensure 2615 / 2615 in M1 ratio is 1
"""
for comp in output_maps["mul_surv"].keys():
    for peak in output_maps["mul_surv"][comp].keys():
        output_maps["mul_surv"][comp][2615]=1
"""
utils.compare_intensity_curves(output_maps["mul_surv"],order,save="plots/relative_intensity/curves_m1_{}.pdf".format(type))
utils.compare_intensity_curves(output_maps["mul2_surv"],order,save="plots/relative_intensity/curves_m2_{}.pdf".format(type))

print(json.dumps(output_maps,indent=1))


### extract the values in the data

data_file = uproot.open(data_path)

counts={}

for spec in specs:
    counts_map_tmp = {}
    for peak in peaks:
        counts_tmp,low_tmp,high_tmp=utils.get_peak_counts(peak,"Tl{}".format(peak),data_file,1,spec=spec,size=7.5)
        err= (low_tmp+high_tmp)/2
        counts_map_tmp[peak]=(counts_tmp,err)
        print("For peak {} counts = {} +/- {}".format(peak,counts_tmp,err))
    counts[spec]=counts_map_tmp

ratios ={}
for peak in counts["mul_surv"]:
    count=counts["mul_surv"][peak]
    count_main =counts["mul_surv"][main]

    ratio = 100*count[0]/count_main[0]
    err_ratio =np.sqrt(ratio*ratio*count_main[1]*count_main[1]/(count_main[0]*count_main[0])+ratio*ratio*count[1]*count[1]/(count[0]*count[0]))
    #if (err_ratio>ratio):
    
    #   err_ratio = ratio+err_ratio
    #ratio=0
    print("Ratio  {} / {} = {} +/- {}".format(peak,main,ratio,err_ratio))
    ratios[peak]=(ratio,err_ratio)

print(json.dumps(ratios,indent=1))

print(json.dumps(output_maps["mul_surv"],indent=1))

for peak in peaks:
    ### create the output we want
    output_plot={}
    for key in output_maps["mul_surv"]:
        output_plot[key]=100*output_maps["mul_surv"][key][peak]
    n="M1"
    utils.plot_relative_intensities(output_plot,ratios[peak][0],ratios[peak][1],order2,"{} keV to {} keV ({})".format(peak,main,n),"plots/relative_intensity/{}_to_{}_{}.pdf".format(peak,main,spec))
    plt.show()
plt.show()


### compare mul_surv to mul2_surv
main = 2204
other=2017
count_2615 = counts["mul_surv"][main]
counts_2615_m2 = counts["mul2_surv"][other]

print("counts M1 = ",count_2615)
print("counts M2 = ",counts_2615_m2)

ratio = 100*counts_2615_m2[0]/count_2615[0]
err_ratio =np.sqrt(ratio*ratio*count_2615[1]*count_2615[1]/(count_2615[0]*count_2615[0])+ratio*ratio*counts_2615_m2[1]*counts_2615_m2[1]/(counts_2615_m2[0]*counts_2615_m2[0]))
print("M2/M1 ratio  ={} +/ - {}".format(ratio,err_ratio))

output_plot={}
peak=2017
for key in output_maps["mul_surv"]:
    output_plot[key]=100*output_maps["mul2_surv"][key][peak]

utils.plot_relative_intensities(output_plot,ratio,err_ratio,order2,"{} kev to {} keV (M2 sum)".format(other,main),"plots/relative_intensity/{}_to_{}_{}.pdf".format(other,main,"M2sum"))
plt.show()
