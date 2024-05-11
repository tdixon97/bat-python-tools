""" plot-activities.py
Python script to plot the activites from the hmixfit fit.
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk)
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
import os
import numpy as np
from IPython.display import display
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
lps.use('legend')
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))

def remove_duplicated(listo,range):
    list_new = []
    range_new=[]
    for item,range_tmp in zip(listo,range):
        if item not in list_new:
            list_new.append(item) 
            range_new.append(range_tmp)
    return list_new,range_new



### parse the arguments
### ---------------------------

parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument("-c","--cfg", type=str,help="Fit cfg file",default="../hmixfit/inputs/cfg/l200a-taup-silver-m1.json")
parser.add_argument("-2","--cfg_2",type=str,help="Second fit cfg file (to make comparison)",default=None)
parser.add_argument("-C","--components",type=str,help="json components config file path",default="cfg/components.json")
parser.add_argument("-o","--obj",type=str,help="obj to plot 'fit_range' 'bi_range' or 'scaling_factor', 'parameter', 'eff_group'",default="fit_range")
parser.add_argument("-t","--type",type=str,help="Type for computation of fit/bi range either 'all', 'M1','M2' or 'by_spec'",default="all")
parser.add_argument("-l","--labels",type=str,help="A list of labels (comma sep.) for the comparison plots",default="fit1,fit2")
parser.add_argument("-O","--outdir",type=str,help="output directory to save plots",default="plots/summary/")
parser.add_argument("-s","--save",type=int,help="save the plots? (1) or print to screen (0)?",default=True)
parser.add_argument("-e","--energy",type=str,help="Energy range to use if the type if eff_groups",default=None)
parser.add_argument("-S","--scale",type=str,help="Scale for the plots (default log)",default="log")
parser.add_argument("-d","--data_band",type=bool,help="Boolean flat to overlay a data band ",default=False)
parser.add_argument("-n","--norm",type=bool,help="Normalise by observed counts in data ",default=False)


types="glob"

### set the parameters
args = parser.parse_args()
scale=args.scale
cfg_path = args.cfg
cfg_path_2=args.cfg_2
json_file=args.components
obj=args.obj
integral_type =args.type
save=args.save
l=utils.csv_string_to_list(args.labels,str)
outdir=args.outdir
priors_file = "cfg/priors.json"
energy=args.energy
data_band = args.data_band
norm = args.norm
idx=0
if (energy is not None):
    energy_low =int(energy.split(",")[0])
    energy_high =int(energy.split(",")[1])


with open(cfg_path,"r") as file:
    cfg =json.load(file)
fit_name,out_dir,specs,ranges,dataset_names,dss=utils.parse_cfg(cfg)
specs,ranges=remove_duplicated(specs,ranges)

pdf_sub_path=""
if ("livetime" in cfg):
    livetime=cfg["livetime"]
else:
    livetime =cfg["fit"]["theoretical-expectations"][dss[idx]][list(cfg["fit"]["theoretical-expectations"][dss[idx]].keys())[0]]["livetime"]
    pdf_sub_path =cfg["fit"]["theoretical-expectations"][dss[idx]][list(cfg["fit"]["theoretical-expectations"][dss[idx]].keys())[0]]["pdf-path"]
print(livetime)
dataset_name=dataset_names[0]
outfile = out_dir+"/hmixfit-"+fit_name+"/analysis.root"
histofile =out_dir+"/hmixfit-"+fit_name+"/histograms.root"
os.makedirs(outdir+"/"+fit_name,exist_ok=True)

do_comp=False
if (cfg_path_2!=None):

    with open(cfg_path_2,"r") as file:
        cfg_2=json.load(file)

    fit_name_2,out_dir_2,specs2,ranges2,dataset_names_2,dss=utils.parse_cfg(cfg_2)
    dataset_name_2=dataset_names_2[0]
    specs2,ranges2=remove_duplicated(specs2,ranges2)
    outfile_2 = out_dir_2+"/hmixfit-"+fit_name_2+"/analysis.root"
    
    do_comp=True

### load the components file
with open(json_file, 'r') as file:
    components = json.load(file, object_pairs_hook=OrderedDict)





### get the fit results as some dataframes
### ---------------------------------------------

with uproot.open(outfile) as data_file:

    trees= utils.get_list_of_not_directories(file  = data_file)
    trees= [element for element in trees if 'counts' in element]

df_tot=utils.get_df_results(trees,dataset_name,specs,outfile)

if (do_comp==True):
    with uproot.open(outfile_2) as data_file:

        trees2= utils.get_list_of_not_directories(file  = data_file)
        trees2= [element for element in trees2 if 'counts' in element]

    df_tot2=utils.get_df_results(trees2,dataset_name_2,specs2,outfile_2)


#### get effiiciencies instead
### -------------------------------------------------------------------------------------

if (obj=="eff_group"):
    det_sel=None
    string_sel=None
    det_type="sum"
    det_types,namet,Ns= utils.get_det_types(det_type,string_sel,det_type_sel=det_sel,level_groups="cfg/level_groups_Sofia.json")
 
    regions ={"peak":[[energy_low,energy_high]],
            "left":[[energy_low-15,energy_low]],
            "right":[[energy_high,energy_high+15]]
            }
    spectrum="mul_surv" 
    pdf_path = cfg["pdf-path"]+pdf_sub_path
    eff_total={}
    for det_name, det_info in det_types.items():
        
        det_list=det_info["names"]
        effs ={}
        for reg in regions:
            effs[reg]={}

        for det,named in zip(det_list,det_info["types"]):
            eff_new,good = utils.get_efficiencies(cfg,spectrum,det,regions,pdf_path,named,spectrum_fit=spectrum,type_fit="icpc")
        
            if (good==1 and (named==det_sel or det_sel==None)):
                effs=utils.sum_effs(effs,eff_new)

        eff_total[det_name]=effs

    for source in eff_total["all"]["peak"]:
        if "outer" in source:
            eff =eff_total["all"]["peak"][source]-(eff_total["all"]["left"][source]+eff_total["all"]["right"][source])*15/(2*(energy_high-energy_low))
            print(f"Eficiency for {source} = {100*eff}")
if (data_band==True):
    if (obj=="eff_group"):
        file = uproot.open(cfg["data-path"]+dss[idx])
        data_counts =utils.get_data_counts_total(spectrum,det_types,regions,file,key_list=["left","right","peak"])
    else:
        raise NotImplementedError("Data band is only implemented for eff_groups option")
#### get the priors
#### ----------------------------------------


with open(priors_file,"r") as file_cfg:
    priors =json.load(file_cfg)

## store the information
### 1) activity quantiles 
### 2) MC pdfs ? i dont think its needed for this script
quantiles={}

for comp in priors["components"]:

    rv1,_,range_ci = utils.extract_prior(comp["prior"])

    point_est = comp["best_fit"]
    low_err = point_est-range_ci[0]
    high_err = range_ci[1]-point_est
    quantiles[comp["name"]]=(point_est,low_err,high_err)
 

### Build a map of what to plot should contain
### 1) the name of the spectrum (all, M1,M2 etc)
### 2) whether its fit_range, bi_range or parameter

list_of_plots=[]

if obj=="scaling_factor" or obj=="parameter":
    list_of_plots=[
        {
            "name":"M1",
            "type":obj
        }

    ]
elif (obj=="fit_range") or (obj=="bi_range"):
    if (integral_type=="all_spec") or (integral_type=="M1") or (integral_type=="M2"):
        list_of_plots.append({"name":integral_type,"type":obj})
    elif (integral_type=="by_spec"):
        for tree in trees:
   
            index = tree.find(dataset_name)
            if index != -1:  
                spec_name = tree[index +1+len(dataset_name):].split(";")[0]

            list_of_plots.append({"name":spec_name,"type":obj})

    else:
        raise ValueError("type must be either all M1, M2 or by_spec")
elif obj=="eff_group":
    for name in eff_total.keys():
        list_of_plots.append({"name":name,"type":obj})
else:
    raise ValueError("obj must be either fit_range, bi_range scaling_factor or parameter")



if (obj=="eff_group"):
    obj=str(regions["peak"][0])

path ="{}/{}/{}_{}_results.pdf".format(outdir,fit_name,obj,integral_type)
if os.path.exists(path):
    os.remove(path)

#### normalise the dataframes
with PdfPages(path) as pdf:
   
    for plot in list_of_plots:

        df =df_tot
        df= df.sort_values(by='comp_name')
        labels=utils.format_latex(np.array(df_tot["comp_name"]))
        names=np.array(df_tot["comp_name"])


        indexs=utils.get_index_by_type(names)
        has_prior=[]
        assay_mean=[]
        assay_low=[]
        assay_high=[]

        ### loop over names
        for name in names:
            if ("prior" in cfg["fit"]["parameters"][name]) and cfg["use-priors"]==True:
                has_prior.append(True)
            else:
                has_prior.append(False)
            
            if (name in quantiles):
                assay_mean.append(quantiles[name][0])
                assay_low.append(quantiles[name][1])
                assay_high.append(quantiles[name][2])
            else:
                assay_mean.append(0)
                assay_low.append(0)
                assay_high.append(0)
        assay_high=np.array(assay_high)
        assay_low=np.array(assay_low)
        assay_mean=np.array(assay_mean)
        
        name = plot["name"]
        type_plot=plot["type"]

        ### data into numpy array
        ### -----------------------
        if (type_plot!="eff_group"):
            x,y,y_low,y_high,assay_norm=utils.get_data_numpy(type_plot,df_tot,name,type=types)
        else:
            x,y,y_low,y_high,assay_norm=utils.get_data_numpy("parameter",df_tot,"M1",type=types)

        if (type_plot=="parameter"):
            y/=31.5
            y_low/=31.5
            y_high/=31.5
            assay_norm/=31.5
        
        if (type_plot=="eff_group"):
            for id,n in enumerate(names):
                eff = (eff_total["all"]["peak"][n]-(eff_total["all"]["left"][n]+eff_total["all"]["right"][n])*15/(2*(energy_high-energy_low)))
                if eff<0:
                    eff=0
                y[id]*=eff
                y_low[id]*=eff
                y_high[id]*=eff
                assay_low[id]*=eff
                assay_high[id]*=eff
                assay_mean[id]*=eff
        
        if (data_band is not False):
            band =utils.get_counts_minuit(np.array([data_counts["all"]["left"],
                                                    data_counts["all"]["peak"],
                                                    data_counts["all"]["right"]]),
                                            np.array([energy_low-15,energy_low,energy_high,energy_high+15])
                            
                                                    )
            band=(band[0]/livetime,band[1]/livetime,band[1]/livetime)
        else:
            band=None

        if (norm):
            y/=band[0]/100
            y_low/=band[0]/100
            y_high/=band[0]/100
            assay_high/=band[0]/100
            assay_low/=band[0]/100
            assay_mean/=band[0]/100
            band=(100,100*band[1]/band[0],100*band[2]/band[0])
            obj="frac"
            name =f"{energy_low} to {energy_high} keV "
        assay_high*=assay_norm
        assay_low*=assay_norm
        assay_mean*=assay_norm

       

        


        if (do_comp==False):
            
            for ids in [np.arange(len(labels)),indexs["U"],indexs["Th"],indexs["K"]]:
                utils.make_error_bar_plot(ids,labels,y,y_low,y_high,data=name,obj=obj,split_priors=False,has_prior=has_prior,
                                        assay_mean=assay_mean,assay_low=assay_low,assay_high=assay_high,scale=scale,data_band=band)

                if (save==True):
                    pdf.savefig()
                else:
                    plt.show()
            
        else:
            if (type_plot!="eff_group"):
                x2,y2,y_low2,y_high2,assay_norm=utils.get_data_numpy(type_plot,df_tot2,name,type=types)
            else:
                x2,y2,y_low2,y_high2,assay_norm=utils.get_data_numpy("parameter",df_tot2,"M1",type=types)

            if (type_plot=="parameter"):
                y2/=31.5
                y_low2/=31.5
                y_high2/=31.5
            if (type_plot=="eff_group"):
                for id,n in enumerate(names):
                    eff = (eff_total["all"]["peak"][n]-(eff_total["all"]["left"][n]+eff_total["all"]["right"][n])*15/(2*(energy_high-energy_low)))
                    if (eff<0):
                        eff=0
                    y2[id]*=eff
                    y_low2[id]*=eff
                    y_high2[id]*=eff

            if (data_band is not False):
                band =utils.get_counts_minuit(np.array([data_counts["all"]["left"],
                                                    data_counts["all"]["peak"],
                                                    data_counts["all"]["right"]]),
                                            np.array([energy_low-15,energy_low,energy_high,energy_high+15])
                            
                                                    )
                band=(band[0]/livetime,band[1]/livetime,band[1]/livetime)
            else:
                band=None

            if (norm):
                y2/=band[0]/100
                y_low2/=band[0]/100
                y_high2/=band[0]/100
                band=(100,100*band[1]/band[0],100*band[2]/band[0])
            
            for ids in [np.arange(len(labels)),indexs["U"],indexs["Th"],indexs["K"]]:
                utils.make_error_bar_plot(ids,labels,y,y_low,y_high,
                                          y2=y2,ylow2=y_low2,yhigh2=y_high2,
                                          data=name,obj=obj,do_comp=True,split_priors=False,has_prior=has_prior,
                                        assay_mean=assay_mean,assay_low=assay_low,assay_high=assay_high,label1=l[0],label2=l[1],data_band=band)

                if (save==True):
                    pdf.savefig()
                else:
                    plt.show()
