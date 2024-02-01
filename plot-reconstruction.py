"""
A script to plot the fit reconstruction of the LEGEND / hmixfit background model
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk) but based on a script from Luigi Pertoldi.
"""

from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')

from collections import OrderedDict
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
from hist import Hist
import hist
import argparse
import os
import sys
import re
import utils
import json
from matplotlib.backends.backend_pdf import PdfPages
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.6,
}



##### the old set of arguments
##### --------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument("-C","--components",type=str,help="json components config file path",default="cfg/components.json")
parser.add_argument("-r","--regions",type=int,help="Shade regions on the plot",default=0)
parser.add_argument("-w","--width",type=int,help="width for canvas",default=7)
parser.add_argument("-a","--save",type=int,help="save the output (1) or print (0)",default=1)
parser.add_argument("-c","--config",type=str,help="hmixfit configuration file for the fit")



#TODO:
#1) make the code figure out fit name automatically
### read the arguments



args = parser.parse_args()
cfg = args.config

with open(cfg,"r") as json_file:
    cfg_dict =json.load(json_file)

### extract all we need from the cfg dict
fit_name,out_dir,det_types,ranges,dataset_name=utils.parse_cfg(cfg_dict)
outfile = out_dir+"/hmixfit-"+fit_name+"/histograms.root"

## could replace with nicer format
type_names=[d.replace("mul_surv","M1").replace("mul2_surv","M2").replace("cat","").replace("/","").replace("sum","s") for d in det_types]



json_file =args.components

exclude_range=0
width=args.width
comp_name = json_file[:-5]
shade_regions=args.regions
save = bool(args.save)
ylowlim =0.01

regions={"$2\\nu\\beta\\beta$":(800,1300),
         "K40":(1455,1465),
         "K42":(1520,1530),         "Tl compton":(1900,2500),
         "Tl peak":(2600,2630)
         }
colors=[vset.red,vset.orange,vset.magenta,vset.teal,"grey"]

get_originals=False

### options to run this code

#### creat the gamma lines plots
### ---------------------------

gamma_line_plots={
    "Tl":
    {
        "lines":[583,2615],
        "data":[],
        "model":[],
        "groups":[0],
        "names":[],
        "count":0
    },
    "Bi":{
        "lines":[609,1764,2204],
        "data":[],
        "model":[],
        "groups":[0],
        "names":[],
        "count":0
    },
    "K":{
        "lines":[1461,1525],
        "data":[],
        "model":[],
        "groups":[0],
        "names":[],
        "count":0
    }

}


with open(json_file, 'r') as file:
    components = json.load(file, object_pairs_hook=OrderedDict)

def get_hist(obj):
    return obj.to_hist()[hist.rebin(bin)]

def get_hist_rb(obj):
    return obj.to_hist()

### save to one PDF
with PdfPages("plots/summary/{}/fit_reconstructions.pdf".format(fit_name)) as pdf:
    
    ## loop over detector type
    for det_type,type_name,fit_range in zip(det_types,type_names,ranges):
        xlow=fit_range[0]
        xhigh=fit_range[1]
        gamma_counter=0


        datasets=["{}_{}".format(dataset_name,det_type.split("/")[1])]
        labels=[type_name]
            
        ## set height of histo
        y=4
        if det_type!="multi":
            y=2.5


        ## create the axes
        if (det_type=="multi" ):
            fig, axes = lps.subplots(len(datasets), 1, figsize=(width, y), sharex=True, gridspec_kw = {'hspace': 0})
        else:
            fig, axes_full = lps.subplots(2, 1, figsize=(width, y), sharex=True, gridspec_kw = { 'height_ratios': [8, 2],"hspace":0})

        if det_type=="multi":
            for ax in axes:
                ax.set_axisbelow(True)
                ax.tick_params(axis="both", which="both", direction="in")
        else:
            axes=np.array([axes_full[0]])
            

        ### create the plot
        with uproot.open(outfile) as f:

            ## loop over datasets
            hs={}
            #for ds, ax, title in zip(datasets, axes.flatten(), labels):
            for i in range(len(datasets)):
                ds = datasets[i]   
                
                if (det_type=="multi"):
                        ax=axes[i]
                        title=labels[i]
                else:
                    ax=axes[0]
                    title=labels[0]
            

                for comp, info in components.items():
                
                    if (det_type!="sum" or ds==datasets[0]):
                    
                        hs[comp]=None

                    ## loop over the contributions to h

                    for name in info["hists"]:


                        if get_originals==True:
                            if name not in f[ds]["originals"]:
                                #raise ValueError("PDF {} not in f[{}]".format(name,ds))
                                continue
                            if hs[comp] is None:
                                hs[comp] = get_hist(f[ds]["originals"][name])
                            else:
                                hs[comp] += get_hist(f[ds]["originals"][name])
                            
                        else:
                            if name not in f[ds]:
                                #raise ValueError("PDF {} not in f[{}]".format(name,ds))
                                continue
                            if hs[comp] is None:
                                hs[comp] = get_hist_rb(f[ds][name])
                            else:
                                hs[comp] += get_hist_rb(f[ds][name])

                    
            
                    if hs[comp] is None:
                        continue
                    
                    ### scale for bin width
                    bin_widths = np.diff(hs[comp].axes.edges[0])
                    
                    if (det_type=="sum" and ds!=datasets[-1]):
                        continue
                    
                    ### make the residual plot
            
                    if (comp=="data"):
                        data=hs[comp].values()
                        bins=hs[comp].axes.centers[0]
                        bin_widths = np.diff(hs[comp].axes.edges[0])
                    if (comp=="total_model"):
                        pred=hs[comp].values()
                        bin_widths = np.diff(hs[comp].axes.edges[0])

                
                    ### rescale bin contents
                    for i in range(hs[comp].size - 2):
                        E=hs[comp].axes.centers[0][i]
                
                        if hs[comp][i] == 0:
                            hs[comp][i] = 1.05*ylowlim

                        ### scale the value to be in units of cts/10 keV
                        if (get_originals==False):
                            hs[comp][i]*=10./bin_widths[i]
                        
                    
                    hs[comp].plot(ax=ax, **style, **info["style"])
                

                ### ----------- save gamma lines info --------

                ### if the peak belongs to a gamma line save it
                low=4000
                high=0
                for i in range(hs["total_model"].size - 2):
                    E=hs["total_model"].axes.centers[0][i]   
                    bw=bin_widths[i]
                    ### get first and last non-0 bin
                   
                    if ((hs["total_model"][i].value*bw/10>1.1*ylowlim) and (E<low)):
                        low=E
                    if ((hs["total_model"][i].value*bw/10>1.1*ylowlim) and (E>high)):
                        high=E
                    if (hs["total_model"][i].value*bw/10<1.1):
                        continue
                    bw = bin_widths[i] 
                    ### loop over different gamma plots

                    for plot_type,gamma_info in gamma_line_plots.items():
                        for gamma_counter in range(len(gamma_info["lines"])):
                            if ( (gamma_counter)<len(gamma_info["lines"]) and (abs(E-gamma_info["lines"][gamma_counter])<bin_widths[i]/2)):
                                
                                gamma_info["model"].append(hs["total_model"][i].value*bw/10)
                                gamma_info["data"].append(hs["data"][i]*(bw)/10)

                                gamma_info["names"].append("{}".format(str(gamma_info["lines"][gamma_counter])))
                                gamma_info["count"]+=1

                for plot_type,gamma_info in gamma_line_plots.items():
                    gamma_info["groups"].append(gamma_info["count"])
                
                if (det_type=="sum" and ds!=datasets[-1]):
                        continue
                

                ### compute residuals
                if (get_originals==True):
                    residual = np.array([((d - m) / (d ** 0.5)) if d > 0.5 else 0 for d, m in zip(data, pred)])
                else:

                    residual =[]
                    for d,m,s in zip(data,pred,bin_widths):
                        obs = d*s/10
                        mu  = m*s/10
                        resid = utils.normalized_poisson_residual(mu,obs)
                        residual.append(resid)
                    residual =np.array(residual)
                
                
                masked_values = np.ma.masked_where((bins > xhigh)  | (bins < xlow), pred)
                masked_values_data = np.ma.masked_where((bins > xhigh)  | (bins < xlow), data)

                maxi = max(masked_values.max(),masked_values_data.max())
                max_y = maxi+2*np.sqrt(maxi)+1

                idx=0

                ### add a shaded region
                if (shade_regions==True):
                    for region in regions.keys():
                        range = regions[region]
                        ax.fill_between(np.array([range[0],range[1]]),np.array([max_y,max_y]),
                                        label=region,alpha=0.3,color=colors[idx],linewidth=0)
                        idx+=1

                ### show some excluded region (eg ROI)
                if (exclude_range!=0):
                    ax.fill_between(np.array([exclude_range[0],exclude_range[1]]),np.array([max_y,max_y]),
                                        label="Not in fit",alpha=0.5,color="grey",linewidth=0)


                ### annotate plot
            
                if ds == datasets[0] or det_type=="sum":
                    legend=ax.legend(loc='upper right',edgecolor="black",frameon=True, facecolor='white',framealpha=1,ncol=1,fontsize=6
                                     )
                    ax.set_legend_annotation()

                    
                elif ds==datasets[len(datasets)-1]:
                    ax.xaxis.set_tick_params(top=False)
                    ax.set_legend_logo(position='upper right', logo_type = 'preliminary', scaling_factor=7)
                else:
                    ax.xaxis.set_tick_params(top=False)

                ### annotate the type of fit
                if (det_type!="multi"):
                    ax.set_legend_logo(position='upper left', logo_type = 'preliminary', scaling_factor=8)
                    
                    ax.annotate("Fit of {} ".format(title),(0.3,0.91),xycoords="axes fraction",fontsize=10)
                    


                ## set labels

                if (det_type=="multi"):
                    ax.set_xlabel("Energy (keV)")
                ax.set_ylabel("Counts / 10 keV")
                ax.set_xlim(xlow,xhigh)
                

                ### now plot the residual
                if (det_type!="multi"):
                    axes_full[1].axhspan(-3,3,color="red",alpha=0.5,linewidth=0)
                    axes_full[1].axhspan(-2,2,color="gold",alpha=0.5,linewidth=0)
                    axes_full[1].axhspan(-1,1,color="green",alpha=0.5,linewidth=0)

                    axes_full[1].errorbar(bins,residual,fmt="o",color="black",markersize=0.8,linewidth=0.6)
                    axes_full[1].set_xlabel("Energy (keV)")
                    axes_full[1].set_ylabel("Residual")
                    axes_full[1].set_yscale("linear")
                    axes_full[1].set_xlim(xlow,xhigh)
                
                    axes_full[1].set_ylim(-6,6)
                    if (exclude_range!=0):
                        axes_full[1].fill_between(np.array([exclude_range[0],exclude_range[1]]),y1=np.array([-6,-6]),y2=np.array([6,6]),
                                    alpha=0.5,color="grey",linewidth=0)
                    axes_full[1].xaxis.set_tick_params(top=False)

                    plt.tight_layout()


        plt.tight_layout()
        for scale in ["log","linear"]:

            if (scale=="log"):
                max_y_p=20*max_y
            else:
                max_y_p=1.3*max_y
                
            ax.set_yscale(scale)
            ax.set_ylim(bottom=ylowlim,top=max_y_p)
            if (save):
                pdf.savefig()
            else:
                plt.show()



    #### make the gamma line plot
    ### ---------------------------------------------------------
    for gamma,gamma_data in gamma_line_plots.items():
            
        gamma_line_data=gamma_data["data"]
        gamma_line_model=gamma_data["model"]
        gamma_index_groups=gamma_data["groups"]
        hist_names=gamma_data["names"]
        fig, axes_full = lps.subplots(2, 1, figsize=(width, y), sharex=True, gridspec_kw = { 'height_ratios': [8, 2],"hspace":0})
        axes=axes_full[0]
        gamma_hist =( Hist.new.Reg(len(gamma_line_data),0,len(gamma_line_data)).Double()
                    )
        gamma_hist_data =( Hist.new.Reg(len(gamma_line_model),0,len(gamma_line_model)).Double()
                    )
        
        for i in range(gamma_hist_data.size-2):
            gamma_hist[i]=gamma_line_model[i]
            gamma_hist_data[i]=gamma_line_data[i]

        ### now make the histos
        gamma_hist_data.plot(ax=axes,**style,color=vset.blue,alpha=0.25,histtype="fill")

        gamma_hist.plot(ax=axes,**style,color="black")
        axes.set_xlim(0,len(gamma_line_data))
        axes.set_ylabel("counts/10 keV")
        axes.set_ylim(0.1,1.2*np.max(gamma_hist.values()))
        axes.set_yscale("linear")
        for x in gamma_index_groups:
            axes.axvline(x=x,linewidth=0.4)

        axes.set_xlabel("")
        axes_full[1].set_xticks(np.arange(len(gamma_line_model))+0.5, hist_names,rotation=90,fontsize=10)
        axes_full[1].set_ylabel("Residual")
        axes_full[1].axhspan(-3,3,color="red",alpha=0.5,linewidth=0)
        axes_full[1].axhspan(-2,2,color="gold",alpha=0.5,linewidth=0)
        axes_full[1].axhspan(-1,1,color="green",alpha=0.5,linewidth=0)

        plt.tight_layout()
        c=0
        for d in type_names:
            if (gamma_index_groups[c]!=gamma_index_groups[c+1]):
                axes.annotate(de, ((1/len(gamma_line_model))*(gamma_index_groups[c]+gamma_index_groups[c+1])/2, 0.91), xycoords="axes fraction", fontsize=6,ha="center")
            c+=1
        ### make a residuals

        data = gamma_hist_data.values()
        pred = gamma_hist.values()
        rs=[]
        for d,p in zip(data,pred):
            rs.append(utils.normalized_poisson_residual(p,d))

        rs=np.array(rs)
        bins = gamma_hist.axes.centers[0]
        axes_full[1].errorbar(bins,rs,fmt="o",color="black",markersize=1,linewidth=1)
        axes_full[1].set_ylim(-max(4,max(abs(rs)))-1,max(4,max(abs(rs)))+1)
        
        if (save):
            pdf.savefig()
        else:
            plt.show()
