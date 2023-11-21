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
parser.add_argument("-c","--cfg", type=str,help="Fit cfg file",default="../hmixfit/inputs/cfg/l200a-taup-silver-m1.json")
parser.add_argument("-o","--out_file",type=str,help="file",default="../hmixfit/results/hmixfit-l200a_taup_silver_dataset_m1_norm-_mcmc.root")
parser.add_argument("-t","--tree_name",type=str,help="tree name",default="l200a_taup_silver_dataset_m1_norm")
parser.add_argument("-s","--spectrum",type=str,help="spectrum",default="raw")
args = parser.parse_args()

outfile=args.out_file
tree_name = args.tree_name
pdf_path = "../hmixfit/inputs/pdfs/l200a-pdfs-v0.3.0-silver/"
data_path ="../hmixfit/inputs/data/l200a-taup-silver-dataset.root"
args = parser.parse_args()

cfg_file = args.cfg
spectrum = args.spectrum
det_type=args.det_type
first_index =utils.find_and_capture(outfile,"hmixfit-l200a_taup_silver_dataset_")
fit_name = outfile[first_index:-10]


with open(cfg_file,"r") as file:
    cfg =json.load(file)

regions={"full": (565,2995),
        "2nu":(800,1300),
         "K peaks":(1400,1600),
         "Tl compton":(1900,2500),
         "Tl peak":(2600,2630)
         }

effs={"full":{},"2nu":{},"K peaks":{},"Tl compton":{},"Tl peak":{}}

if (det_type!="sum"):
    effs=utils.get_efficiencies(cfg,spectrum,det_type,regions,pdf_path)
else:
    det_types=["icpc","bege","ppc","coax"]
    for d in det_types:
    
        effs = utils.sum_effs(effs,utils.get_efficiencies(cfg,spectrum,d,regions,pdf_path))

print(json.dumps(effs,indent=1))

### now open the MCMC file

tree= "{}_mcmc".format(tree_name)
df =utils.ttree2df(outfile,tree)
print("Read dataframe")
df=df.query("Phase==1")
print("Quried to get Phase==1")

df=df.drop(columns=['Chain','Iteration','Phase','LogProbability','LogLikelihood','LogPrior'])

print("Start computing sums")
### compute the sums of each column weighted by efficiiency 
sums_full={}
for key,eff in effs.items():
    df_tmp= df.copy()
    
    eff_values = np.array(list(eff.values()))
    eff_columns = list(eff.keys())
    df_tmp[eff_columns] *= eff_values[np.newaxis,:]
    
    sums=np.array(df_tmp.sum(axis=1))
    sums_full[key]=sums

### get the correspondong counts in data
file = uproot.open(data_path)
time=0.1273

data_counts={"full":0,"2nu":0,"Tl compton":0,"Tl peak":0,"K peaks":0}

if (det_type!="sum"):
    data_counts=utils.get_data_counts(spectrum,det_type,regions,file)
else:
    det_types=["icpc","bege","ppc","coax"]
    for d in det_types:
        data_counts = utils.sum_effs(data_counts,utils.get_data_counts(spectrum,d,regions,file))




fig, axes_full = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})

for key in effs.keys():
    fig, axes_full = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})

    data = sums_full[key]*time
    data_real = np.random.poisson(data)
    range = (int(min(np.min(data_real),data_counts[key]))-0.5,int(max(np.max(data_real),data_counts[key]))+0.5)

    bins = int(range[1]-range[0])
    axes_full.hist(data,range=range,bins=bins,alpha=0.3,color=vset.blue,label="Estimated parameter")
    axes_full.hist(data_real,range=range,bins=bins,alpha=0.3,color=vset.red,label="Expected realisations")

    axes_full.set_xlabel("counts")
    axes_full.set_ylabel("Prob [arb]")
    axes_full.plot(np.array([data_counts[key],data_counts[key]]),np.array([axes_full.get_ylim()[0],axes_full.get_ylim()[1]]),label="Data",color="black")
    axes_full.set_title("Model reconstruction for {}".format(key))
    plt.legend()
    plt.savefig("plots/region_counts/{}_{}_{}.pdf".format(det_type,key,fit_name))

plt.show()

