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

priors_file ="priors.json"
pdf_path = "../hmixfit/inputs/pdfs/l200a-pdfs_vancouver23/"
data_path="../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v0.1.root"
spec="mul_surv"
with open(priors_file,"r") as file_cfg:
    priors =json.load(file_cfg)

livetime=0.36455
order=["fiber_shroud","sipm","birds_nest","wls_reflector","pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud"]

### we need to extract the PDFs and get the counts in relevant gamma lines first

type="Bi212Tl208"
type="Pb214Bi214"
if (type=="Bi212Tl208"):
    peaks=[583,727,861,1593,2104,2615]
    main=2615
elif(type=="Pb214Bi214"):
    peaks=[609,1238,1378,1764,2204,2447]
    main=1764



output_map ={}
with PdfPages('plots/relative_intensity/pdfs_{}.pdf'.format(type)) as pdf:
    
    for comp in priors["components"]:
    
        output_tmp ={}
        if (type not in comp["name"]):
            continue

        for peak in peaks:

            pdf_tmp,N = utils.get_pdf_and_norm(pdf_path+comp["file"],b=10,r=(peak-15,peak+15))
            center= pdf_tmp[1]
            bkg = (pdf_tmp[2]+pdf_tmp[0])/2.
            value = center-bkg
            if (value<0):
                value=0
            output_tmp[peak]=value
            utils.plot_mc(pdf_tmp,"Peak of {} in {}".format(peak,comp["name"]),pdf,range_x=(peak-15,peak+15),range_y=(1,center*2),scale="linear")

        output_map[comp["name"][11:]]=output_tmp

for comp in output_map.keys():
    norm  = output_map[comp][main]
    for peak in output_map[comp].keys():
        output_map[comp][peak]/=norm
utils.compare_intensity_curves(output_map,order,save="plots/relative_intensity/curves_{}.pdf".format(type))



### extract the values in the data
data_path="../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v0.1.root"

data_file = uproot.open(data_path)

counts={}
errs=[]
for peak in peaks:
    counts_tmp,low_tmp,high_tmp=utils.get_peak_counts(peak,"Tl{}".format(peak),data_file,1)
    err= (low_tmp+high_tmp)/2
    counts[peak]=(counts_tmp,err)



for peak in counts:
    count=counts[peak]
    count_main =counts[main]

    ratio = 100*count[0]/count_main[0]
    err_ratio =np.sqrt(ratio*ratio*count_main[1]*count_main[1]/(count_main[0]*count_main[0])+ratio*ratio*count[1]*count[1]/(count[0]*count[0]))
    print("Ratio  {} / {} = {} +/- {}".format(peak,main,ratio,err_ratio))


