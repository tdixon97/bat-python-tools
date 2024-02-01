"""
A script to plot the fit projections. 
Author: toby dixon (toby.dixon.23@ucl.ac.uk)
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
import sys
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

### parse the arguments
### --------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument('-d','--det_type', type=str, help='which type of splitting to use for the data, either "types,string","chan" or "floor"',default='chan')
parser.add_argument('-e','--data_sel', type=str, help='Detector type to use default None',default=None)
parser.add_argument('-S','--string_sel', type=int, help='string to select default None',default=None)
parser.add_argument("-c","--cfg", type=str,help="Fit cfg file",default="../hmixfit/inputs/cfg/l200a-taup-silver-m1.json")
parser.add_argument("-s","--spectrum",type=str,help="spectrum",default="mul_surv")
parser.add_argument("-a","--save",type=int,help="Bool to save the plots ",default=1)

args = parser.parse_args()


### parse the args
cfg_file = args.cfg

save = args.save
spectrum = args.spectrum
det_type=args.det_type

det_sel=args.data_sel
string_sel=args.string_sel



### get a list of detector types to consider
### -----------------------------------------
det_types,namet,Ns= utils.get_det_types(det_type,string_sel,det_type_sel=det_sel,level_groups="level_groups_Sofia.json")

with open(cfg_file,"r") as file:
    cfg =json.load(file)

### extract all we need from the cfg dict
fit_name,out_dir,_,_,dataset_name=utils.parse_cfg(cfg)
outfile = out_dir+"/hmixfit-"+fit_name+"/mcmc_small.root"
pdf_path = cfg["pdf-path"]
data_path=cfg["data-path"]+dataset_name

### list of regions
with open("cfg/regions.json","r") as file:
    regions =json.load(file)



###
### creat the efficiency maps (per dataset)
### ---------------------------------------
eff_total={}
for det_name, det_info in det_types.items():
    
    det_list=det_info["names"]
    effs ={}
    for reg in regions:
        effs[reg]={}

    for det,named in zip(det_list,det_info["types"]):
        eff_new,good = utils.get_efficiencies(cfg,spectrum,det,regions,pdf_path,named,spectrum_fit="mul_surv",type_fit="icpc")
       
        if (good==1 and (named==det_sel or det_sel==None)):
            effs=utils.sum_effs(effs,eff_new)

    eff_total[det_name]=effs


### now open the MCMC file
tree= "{}_mcmc".format(fit_name)
df =utils.ttree2df(outfile,"mcmc","Phase==1",N=2000000)

df=df.query("Phase==1")
df=df.drop(columns=['Chain','Iteration','Phase','LogProbability','LogLikelihood','LogPrior'])

### compute the sums of each column weighted by efficiiency 
### -------------------------------------------------------

start_time = time.time()
sums_total={}
groups=["all","K42","K40","Bi212Tl208","Pb214Bi214","alpha","Co60","2vbb"]


## loop over spectra
## --------------------------------
for dataset,effs in eff_total.items():

    sums_full={}

    for group in groups:
        sums_full[group]={}

    for key,eff in effs.items():

        for group in groups:

            df_tmp= df.copy()
            
            ### filter
            if (group!="all"):
                df_tmp=df_tmp.filter(regex=group)
                filtered_columns =df_tmp.columns
                eff_tmp = {key_e: value_e for key_e, value_e in eff.items() if key_e in filtered_columns}
            else:
                eff_tmp=eff
            eff_values = np.array(list(eff_tmp.values()))
            eff_columns = list(eff_tmp.keys())
            
            df_tmp[eff_columns] *= eff_values[np.newaxis,:]
            sums=np.array(df_tmp.sum(axis=1))
            sums_full[group][key]=sums


    sums_total[dataset]=sums_full

end_time = time.time()

#### produce a plot of the reconstructed number of counts
### ---------------------------------------------------
if (det_type=="all"):
    colors={"alpha":"orange","2vbb":"#999933","all":"#000000","Bi212Tl208":"#33BBEE","Pb214Bi214":"#009988","K42":"#EE3377","K40":"#CC3311","Co60":vset.orange}
    with PdfPages("plots/summary/{}/ROI_breakdown.pdf".format(fit_name)) as pdf:
        
        roi_map={}
        ### loop over the detector groups
        for det_name in det_types:
            roi_map[det_name]={}

            for region in ["ROI","ROI_no_cut"]:
                roi_map[det_name][region]={}
                fig, axes_full = lps.subplots(1, 1,figsize=(4, 3), sharex=True, gridspec_kw = { "hspace":0})
                maxi=0
                for group in groups:

                    sums= sums_total[det_name][group][region]*cfg["livetime"]
                    maxi = 0
                    if (group!="all" and np.max(sums)*1.1 >maxi):
                        rangef=(0,np.max(sums)*1.1)                
                        bins = 500
                    else:
                        rangef =(0,maxi)

                    if (group!="all"):
                        axes_full.hist(sums,range=rangef,bins=bins,alpha=0.3,color=colors[group],label=utils.format_latex([group])[0])
                    if np.max(sums)*1.1>maxi:
                        maxi=np.max(sums)*1.1
                   
                    h,b = np.histogram(sums, bins=500,range=rangef ,density=True)
                    mode = b[np.argmax(h)]
                    quantiles =np.quantile(sums,[.16,.84])
                    roi_map[det_name][region][group]={"mode":mode,"q16":quantiles[0],"q84":quantiles[1]}
                    ### get mode and quantiles
                    
                    axes_full.set_xlim(0,maxi)
                    axes_full.set_xlabel("Reconstructed counts")
                    axes_full.set_ylabel("Prob. [arb]")
                    axes_full.set_title("Model reconstruction for {} {}".format(det_name,region))
                plt.legend(fontsize=8,ncol=2,loc="upper right")
                
                if (save):
                    pdf.savefig()
                else:
                    plt.show()

        with open("plots/summary/{}/roi_bkg.json".format(fit_name),"w") as json_file:
            json.dumps(roi_map,json_file,indent=1)

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

### get the correspondong counts in data
### ----------------------------------------------------------

file = uproot.open(data_path)
time=cfg["livetime"]
data_counts_total={}

## loop over datasets
for det_name,det_info in det_types.items():

    det_list=det_info["names"]
    dt=det_info["types"]
    data_counts={}

    for reg in regions.keys():
        data_counts[reg]=0

    for det,type in zip(det_list,dt):
        if (type==det_sel or det_sel==None):
            data_counts = utils.sum_effs(data_counts,utils.get_data_counts(spectrum,det,regions,file))

    data_counts_total[det_name]=data_counts

summary={}
for reg in regions.keys():
    summary[reg]={}


#### plot the model prediction vs data in the regions
#### -------------------------------------------------------------

with PdfPages("plots/summary/{}/observables_{}.pdf".format(fit_name,det_type)) as pdf:

    for det_name in det_types:
        
        effs = eff_total[det_name]
        data_counts =data_counts_total[det_name]
        sums_full = sums_total[det_name]["all"]

        exposure = det_types[det_name]["exposure"]
        for key in effs.keys():
            fig, axes_full = lps.subplots(1, 1,figsize=(4, 3), sharex=True, gridspec_kw = { "hspace":0})

            data = sums_full[key]*time
            if (data[0]>1E7):
                print("Too many counts for {}".format(key))
                data*=0
                data_real*=0
            med = np.percentile(data,50)
            data_real = np.random.poisson(np.ones(len(data))*med)

            low = np.percentile(data,16)
            high =np.percentile(data,50+34)
            high-=med
            low = med-low
            summary[key][det_name]=[low,med,high,exposure]

            rangef = (int(min(np.min(data_real),data_counts[key]))-0.5,int(max(np.max(data_real),data_counts[key]))+0.5)
        
            bins = 100
            
            
            axes_full.hist(data,range=rangef,bins=bins,alpha=0.3,color=vset.blue,label="Estimated parameter")
            axes_full.hist(data_real,range=rangef,bins=bins,alpha=0.3,color=vset.red,label="Expected realisations")

            axes_full.set_xlabel("counts")
            axes_full.set_ylabel("Prob [arb]")
            axes_full.plot(np.array([data_counts[key],data_counts[key]]),np.array([axes_full.get_ylim()[0],axes_full.get_ylim()[1]]),label="Data",color="black")
            axes_full.set_title("Model reconstruction for {} {:.2g}$^{{+{:.2g}}}_{{-{:.2g}}}$".format(key,med,high,low))
            plt.legend(fontsize=8)

            if (save==True):    
                pdf.savefig()
            else:
                plt.show()


### now create the summary plots
### ---------------------------------------------------------------
with PdfPages("plots/summary/{}/projections_{}.pdf".format(fit_name,det_type)) as pdf:
    for region in summary.keys():
        fig, axes_full = lps.subplots(2, 1, figsize=(5, 3), sharex=True, gridspec_kw = { 'height_ratios': [8, 2],"hspace":0})

        
        # loop over dataspectra
        lows=[]
        highs=[]
        meds=[]
        datas=[]
        names=[]
        exposures=[]
        for det in summary[region].keys():
            params = summary[region][det]

            low =params[1]-params[0]
            high = params[1]+params[2]
            med = params[1]
            exposure = params[3]

            lows.append(low)
            highs.append(high)
            meds.append(med)
            datas.append(data_counts_total[det][region])
            names.append(det)
            exposures.append(exposure)
            
        ## create the x and y array for fill between
        xs=[0,1]
        hlow=[lows[0]/exposures[0],lows[0]/exposures[0]]
        hhigh=[highs[0]/exposures[0],highs[0]/exposures[0]]
        data=[datas[0]/exposures[0],datas[0]/exposures[0]]
        med=[meds[0]/exposures[0],meds[0]/exposures[0]]

        names=np.array(names)

        for i in range(1,len(names)):
            xs.extend([i,i+1])

            hlow.extend([lows[i]/exposures[i],lows[i]/exposures[i]])
            hhigh.extend([highs[i]/exposures[i],highs[i]/exposures[i]])
            data.extend([datas[i]/exposures[i],datas[i]/exposures[i]])
            med.extend([meds[i]/exposures[i],meds[i]/exposures[i]])

        axes_full[0].fill_between(xs,data,color=vset.blue,label="Data",alpha=0.3)

        axes_full[0].plot(xs,med,color=vset.red,label="Best fit")
        for N in Ns:
            axes_full[0].axvline(x=N)
        axes_full[0].set_xlabel("Detector type")
        axes_full[0].set_ylabel("Counts/kg-yr")
        reg_fancy=region
        if region=="Tlpeak":
            reg_fancy = "$^{208}$Tl 2615 keV peak"
        if region=="Bipeak":
            reg_fancy="$^{214}$Bi 1764 keV peak"
        if region=="K42":
            reg_fancy="$^{42}$K 1520 keV peak"
        if region=="K40":
            reg_fancy="$^{40}$K 1461 keV peak"
        if ("ROI" in region):
            reg_fancy ="bkg index region"
        axes_full[0].set_title("Counts per group for {}".format(reg_fancy))
        if (det_type!="chan"):
            axes_full[0].set_xticks(0.5+np.arange(len(names)),names,rotation=80,fontsize=0)
        axes_full[0].set_ylim(0,max(max(data),max(hhigh))*1.5)
        axes_full[0].legend(loc="upper right")
        axes_full[0].set_yscale("linear")

        ## make residual plot
        
        for d,m,c in zip(datas,meds,names):
            if ((d>0)and (m==0)):
                print("For {} there are data events but not MC ".format(c))
            if ((d==0) and (m>0)):
                print("For {} there are MC events but notdata ".format(c))

        residual =[]
        for d,m in zip(datas,meds):
            obs = d
            mu  = m
            resid = utils.normalized_poisson_residual(mu,obs)
            residual.append(resid)
        residual =np.array(residual)
        name_plot=names
        if (det_type=="chan"):
            for i in range(len(names)):
                meta = LegendMetadata()
                name_tmp = utils.number2name(meta,names[i])
            
                name_plot[i]=name_tmp
            
        axes_full[1].errorbar(0.5+np.arange(len(names)),residual,yerr=np.zeros(len(residual)),fmt="o",color="black",markersize=2)
        axes_full[1].axhspan(-3,3,color="red",alpha=0.5,linewidth=0)
        axes_full[1].axhspan(-2,2,color="gold",alpha=0.5,linewidth=0)
        axes_full[1].axhspan(-1,1,color="green",alpha=0.5,linewidth=0)
        axes_full[1].set_ylabel("Residual")
        axes_full[1].set_yscale("linear")
        axes_full[1].set_xlim(0,max(xs))
        axes_full[1].set_ylim(min(residual)-2,max(residual+2))
        fontsize=12
        if (det_type=="chan"):
            fontsize=4
        if (det_type!="chan"):    
            axes_full[1].set_xticks(0.5+np.arange(len(name_plot)),name_plot,rotation=80,fontsize=fontsize)

        plt.tight_layout()

        if (save==True):
            pdf.savefig()
        else:
            plt.show()
