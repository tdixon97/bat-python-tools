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
print(fit_name)

with open(cfg_file,"r") as file:
    cfg =json.load(file)


icpc_comp_list=cfg["fit"]["theoretical-expectations"]["l200a-taup-silver-dataset.root"]["{}/{}".format(spectrum,det_type)]["components"]
effs={"full":{},"2nu":{},"K peaks":{},"Tl compton":{},"Tl peak":{}}

regions={"full": (565,2995),
        "2nu":(800,1300),
         "K peaks":(1400,1600),
         "Tl compton":(1900,2500),
         "Tl peak":(2600,2630)
         }

for key,region in regions.items():
    effs[key]["2vbb_bege"]=0
    effs[key]["2vbb_coax"]=0
    effs[key]["2vbb_ppc"]=0
    effs[key]["2vbb_icpc"]=0


for comp in icpc_comp_list:
    print("File = {}".format(comp["root-file"]))
    for key in comp["components"].keys():
      
        par = key

    print("par = {} ".format(par))

    ## now open the file

    file = uproot.open(pdf_path+comp["root-file"])
   
    hist = file["{}/{}".format(spectrum,det_type)]
    N  = int(file["number_of_primaries"])

    ## use pyroot for the integral
    hist = hist.to_hist()

    for key,region in regions.items():

        eff= float(utils.integrate_hist(hist,region[0],region[1]))

        effs[key][par]=eff/N
   
    


### now open the MCMC file

tree= "{}_mcmc".format(tree_name)
df =utils.ttree2df(outfile,tree)
df=df.query("Phase==1")


df=df.drop(columns=['Chain','Iteration','Phase','LogProbability','LogLikelihood','LogPrior'])
sums_full={}
for key,eff in effs.items():
    df_tmp= df.copy()
    
    for source,effi in eff.items():
        df_tmp[source] *= effi
    
    sums=np.array(df_tmp.sum(axis=1))

    sums_full[key]=sums



### get the correspondong counts in data
file = uproot.open(data_path)

hist =file["{}/{}".format(spectrum,det_type)]
data_counts={}
time=0.1273
hist=hist.to_hist()
for region,range in regions.items():
    data= float(utils.integrate_hist(hist,range[0],range[1]))
    data=data/time

    data_counts[region]=data

fig, axes_full = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})

for key in effs.keys():
    fig, axes_full = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})

    data = sums_full[key]
    data_real = np.random.poisson(data)
    axes_full.hist(sums_full[key],range=(min(data),max(data)),bins=100,alpha=0.3,color=vset.blue,label="Estimated parameter")
    axes_full.hist(data_real,range=(min(data),max(data)),bins=100,alpha=0.3,color=vset.red,label="Expected realisations")

    axes_full.set_xlabel("counts/yr")
    axes_full.set_ylabel("Prob [arb]")
    axes_full.plot(np.array([data_counts[key],data_counts[key]]),np.array([axes_full.get_ylim()[0],axes_full.get_ylim()[1]]),label="Data",color="black")
    axes_full.set_title("Model reconstruction for {}".format(key))
    plt.legend()
    plt.savefig("plots/region_counts/{}_{}_{}.pdf".format(det_type,key,fit_name))

plt.show()

