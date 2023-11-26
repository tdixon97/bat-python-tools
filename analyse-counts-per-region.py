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


parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument('-d','--det_type', type=str, help='Detector type',default='icpc')
parser.add_argument("-p","--pdf", type=str,help="Path to the PDFs")
parser.add_argument("-a","--data", type=str,help="Path to the data")
parser.add_argument('-e','--data_sel', type=str, help='Detector type',default=None)
parser.add_argument('-S','--string_sel', type=int, help='string to select',default=0)
parser.add_argument("-c","--cfg", type=str,help="Fit cfg file",default="../hmixfit/inputs/cfg/l200a-taup-silver-m1.json")
parser.add_argument("-o","--out_file",type=str,help="file",default="../hmixfit/results/hmixfit-l200a_taup_silver_dataset_m1_norm-_mcmc.root")
parser.add_argument("-t","--tree_name",type=str,help="tree name",default="l200a_taup_silver_dataset_m1_norm")
parser.add_argument("-s","--spectrum",type=str,help="spectrum",default="raw")
args = parser.parse_args()

outfile=args.out_file
tree_name = args.tree_name
pdf_path =args.pdf
data_path =args.data
args = parser.parse_args()

cfg_file = args.cfg
spectrum = args.spectrum
det_type=args.det_type
first_index =utils.find_and_capture(outfile,"hmixfit-l200a_")
fit_name = outfile[first_index:-10]
det_sel=args.data_sel
string_sel=args.string_sel
print(det_sel)
### get a list of detector types to consider
det_types,namet= utils.get_det_types(det_type,string_sel,det_sel)
print(json.dumps(det_types,indent=1))

with open(cfg_file,"r") as file:
    cfg =json.load(file)

regions={"full": (565,2995),
        "2nu":(800,1300),
         "K40":(1455,1465),
         "K42":(1520,1530),
         "Tl compton":(1900,2500),
         "Tl peak":(2605,2625)
         }
eff_total={}
### creat the efficiency maps (per dataset)
for det_name, det_info in det_types.items():
    
    det_list=det_info["names"]

    effs={"full":{},"2nu":{},"K40":{},"K42":{},"Tl compton":{},"Tl peak":{}}

    for det,named in zip(det_list,det_info["types"]):
        eff_new,good = utils.get_efficiencies(cfg,spectrum,det,regions,pdf_path,named,"mul_surv")
        print(named)
        if (good==1 and (named==det_sel or det_sel=="all")):
            effs=utils.sum_effs(effs,eff_new)

    eff_total[det_name]=effs

### now open the MCMC file
tree= "{}_mcmc".format(tree_name)
df =utils.ttree2df(outfile,"mcmc")
df=df.query("Phase==1").iloc[0:500000]
df=df.drop(columns=['Chain','Iteration','Phase','LogProbability','LogLikelihood','LogPrior'])



### compute the sums of each column weighted by efficiiency 
start_time = time.time()
sums_total={}
## loop over spectra
for dataset,effs in eff_total.items():

    sums_full={}
    for key,eff in effs.items():
        df_tmp= df.copy()
        
        eff_values = np.array(list(eff.values()))
        eff_columns = list(eff.keys())
        df_tmp[eff_columns] *= eff_values[np.newaxis,:]
        
        sums=np.array(df_tmp.sum(axis=1))
        sums_full[key]=sums

    sums_total[dataset]=sums_full
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

### get the correspondong counts in data
file = uproot.open(data_path)
time=cfg["livetime"]
data_counts_total={}

## loop over datasets
for det_name,det_info in det_types.items():

    det_list=det_info["names"]
    dt=det_info["types"]
    data_counts={"full":0,"2nu":0,"Tl compton":0,"Tl peak":0,"K40":0,"K42":0}
    for det,type in zip(det_list,dt):
        if (type==det_sel or det_sel=="all"):
            data_counts = utils.sum_effs(data_counts,utils.get_data_counts(spectrum,det,regions,file))

    data_counts_total[det_name]=data_counts


summary={"full":{},"2nu":{},"Tl compton":{},"Tl peak":{},"K40":{},"K42":{}}


### loop over datasets
for det_name in det_types:
    
    effs = eff_total[det_name]
    data_counts =data_counts_total[det_name]
    sums_full = sums_total[det_name]
    
    for key in effs.keys():
        fig, axes_full = lps.subplots(1, 1,figsize=(5, 4), sharex=True, gridspec_kw = { "hspace":0})

        data = sums_full[key]*time
        data_real = np.random.poisson(data)
        if (data[0]>1E7):
            print("Too many counts for {}".format(key))
            data*=0
            data_real*=0
        med = np.percentile(data,50)
        low = np.percentile(data,16)
        high =np.percentile(data,50+34)
        high-=med
        low = med-low
        summary[key][det_name]=[low,med,high]

        rangef = (int(min(np.min(data_real),data_counts[key]))-0.5,int(max(np.max(data_real),data_counts[key]))+0.5)
      
        bins = int(rangef[1]-rangef[0])
        
        
        axes_full.hist(data,range=rangef,bins=bins,alpha=0.3,color=vset.blue,label="Estimated parameter")
        axes_full.hist(data_real,range=rangef,bins=bins,alpha=0.3,color=vset.red,label="Expected realisations")

        axes_full.set_xlabel("counts")
        axes_full.set_ylabel("Prob [arb]")
        axes_full.plot(np.array([data_counts[key],data_counts[key]]),np.array([axes_full.get_ylim()[0],axes_full.get_ylim()[1]]),label="Data",color="black")
        axes_full.set_title("Model reconstruction for {} {:.2g}$^{{+{:.2g}}}_{{-{:.2g}}}$".format(key,med,high,low))
        plt.legend()
        plt.savefig("plots/region_counts/counts/count_{}_{}_{}.pdf".format(det_name,key,fit_name))

        plt.close()



## now create the summary plots

for region in summary.keys():
    fig, axes_full = lps.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw = { 'height_ratios': [8, 2],"hspace":0})

    
    # loop over dataspectra
    lows=[]
    highs=[]
    meds=[]
    datas=[]
    names=[]
    for det in summary[region].keys():
        params = summary[region][det]

        low =params[1]-params[0]
        high = params[1]+params[2]
        med = params[1]

        lows.append(low)
        highs.append(high)
        meds.append(med)
        datas.append(data_counts_total[det][region])
        names.append(det)
        
    ## create the x and y array for fill between
    xs=[0,1]
    hlow=[lows[0],lows[0]]
    hhigh=[highs[0],highs[0]]
    data=[datas[0],datas[0]]
    med=[meds[0],meds[0]]

    names=np.array(names)

    for i in range(1,len(names)):
        xs.extend([i,i+1])
        hlow.extend([lows[i],lows[i]])
        hhigh.extend([highs[i],highs[i]])
        data.extend([datas[i],datas[i]])
        med.extend([meds[i],meds[i]])

    axes_full[0].fill_between(xs,data,color=vset.blue,label="Data",alpha=0.3)

    #axes_full.fill_between(xs,hlow,hhigh,color=vset.teal,alpha=0.3,label="Model prediction")
    axes_full[0].plot(xs,med,color=vset.red,label="Best fit")

    axes_full[0].set_xlabel("Detector type")
    axes_full[0].set_ylabel("Counts")
    axes_full[0].set_title("Counts per det type for {}".format(region))
    axes_full[0].set_xticks(0.5+np.arange(len(names)),names,rotation=80,fontsize=6)
    axes_full[0].set_ylim(0,max(max(data),max(hhigh))*1.1)
    axes_full[0].legend(loc="upper right")
    axes_full[0].set_yscale("linear")

    ## make residual plot
    
    for d,m,c in zip(datas,meds,names):
        if ((d>0)and (m==0)):
            print("For {} there are data events but not MC ".format(c))
        if ((d==0) and (m>0)):
            print("For {} there are MC events but notdata ".format(c))

    residual = np.array([((d - m) / (m ** 0.5)) if m>0 else 0 for d, m in zip(datas, meds)])
    name_plot=names
    if (det_type=="chan"):
        for i in range(len(names)):
            name_tmp = utils.number2name(meta,names[i])

            name_plot[i]=name_tmp
        
    axes_full[1].errorbar(0.5+np.arange(len(names)),residual,yerr=np.ones(len(residual)),fmt="o",color=vset.blue,markersize=2)
    axes_full[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes_full[1].set_ylabel("Residual")
    axes_full[1].set_yscale("linear")
    axes_full[1].set_xlim(0,max(xs))
    axes_full[1].set_ylim(min(residual)-2,max(residual+2))
    fontsize=10
    if (det_type=="chan"):
        fontsize=4

    axes_full[1].set_xticks(0.5+np.arange(len(name_plot)),name_plot,rotation=80,fontsize=fontsize)

    plt.tight_layout()

    plt.savefig("plots/region_counts/summary_{}_{}_{}_{}_string_{}.pdf".format(region,namet,spectrum,det_sel,string_sel))
plt.show()

