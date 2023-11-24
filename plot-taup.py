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
parser.add_argument('-d','--det_type', type=str, help='Detector type',default='all')
parser.add_argument('-l','--lower_x', type=int, help='Low range',default=565)
parser.add_argument('-u','--upper_x', type=int, help='Upper range',default=3000)
parser.add_argument("-s","--scale",type=str,help="Scale",default="log")
parser.add_argument("-o","--out_file",type=str,help="file",default="../hmixfit/results/hmixfit-l200a-taup-silver-dataset-m1-histograms.root")
parser.add_argument("-c","--components",type=str,help="json components config file path",default="components.json")
parser.add_argument("-b","--bins",type=int,help="Binning",default=15)
parser.add_argument("-r","--regions",type=int,help="Shade regions",default=0)
parser.add_argument("-f","--fit_name",type=str,help="fit name",default="l200a_vancouver23_dataset_v0_1")


### read the arguments


args = parser.parse_args()
det_type=args.det_type
xlow=args.lower_x
xhigh=args.upper_x
ylowlim =0.05
scale=args.scale
outfile=args.out_file
first_index =utils.find_and_capture(outfile,"hmixfit-")
name_out = outfile[first_index:-16]
fit_name = args.fit_name
json_file =args.components
bin = args.bins
exclude_range=(1455,1540)
comp_name = json_file[:-5]
shade_regions=args.regions

regions={"$2\\nu\\beta\\beta$":(800,1300),
         "K peaks":(1400,1600),
         "Tl compton":(1900,2500),
         "Tl peak":(2600,2630)
         }
colors=[vset.red,vset.orange,vset.magenta,vset.teal]


with open(json_file, 'r') as file:
    components = json.load(file, object_pairs_hook=OrderedDict)

def get_hist(obj):
    return obj.to_hist()[132:4195][hist.rebin(bin)]

def get_hist_rb(obj):
    return obj.to_hist()

if det_type=="multi" or det_type =="sum":
    datasets = ["{}_bege".format(fit_name), "{}_icpc".format(fit_name),"{}_ppc".format(fit_name),"{}_coax".format(fit_name)]
    labels = ["BEGe detectors after QC", "ICPC detectors after QC","PPC detectors after QC","COAX detectors after QC"]
else:
    datasets=["{}_{}".format(fit_name,det_type)]
    labels=["{} detectors after QC".format(det_type)]

y=6
if det_type!="multi":
    y=4


## create the axes
if (det_type=="multi" ):
    fig, axes = lps.subplots(len(datasets), 1, figsize=(6, y), sharex=True, gridspec_kw = {'hspace': 0})
else:
    fig, axes_full = lps.subplots(2, 1, figsize=(6, y), sharex=True, gridspec_kw = { 'height_ratios': [8, 2],"hspace":0})

if det_type=="multi":
    for ax in axes:
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", direction="in")
else:
    axes=np.array([axes_full[0]])
    
get_originals=False






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


                if get_originals=="True":
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
            

            ### rescale bin contents
            for i in range(hs[comp].size - 2):
                E=hs[comp].axes.centers[0][i]
        
                if hs[comp][i] == 0:
                
                    hs[comp][i] = 1.05*ylowlim
                if (get_originals==False):
                    hs[comp][i]*=10/bin_widths[i]
            
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

          
            hs[comp].plot(ax=ax, **style, **info["style"])

           
        if (det_type=="sum" and ds!=datasets[-1]):
                continue
        
        if (get_originals==False):
            residual = np.array([((d - m) / (m ** 0.5)) if d > -1 else 0 for d, m in zip(data, pred)])
        else:
            residual = np.array([((d*s/10 - m*s/10) / ((m*s/10) ** 0.5)) if d > -1 else 0 for d, m,s in zip(data, pred,bin_widths)])

        
        masked_values = np.ma.masked_where((bins > xhigh)  | (bins < xlow), pred)
        masked_values_data = np.ma.masked_where((bins > xhigh)  | (bins < xlow), data)

        maxi = max(masked_values.max(),masked_values_data.max())
        max_y = maxi+2*np.sqrt(maxi)+1
        if (scale=="log"):
            max_y=3*max_y
        else:
            max_y=1.3*max_y
            
        ax.set_yscale("linear")
        ax.set_ylim(bottom=ylowlim,top=max_y)
        idx=0
        if (shade_regions==True):
            for region in regions.keys():
                range = regions[region]
                ax.fill_between(np.array([range[0],range[1]]),np.array([max_y,max_y]),
                                label=region,alpha=0.3,color=colors[idx])
                idx+=1

        if (exclude_range!=0):
            ax.fill_between(np.array([exclude_range[0],exclude_range[1]]),np.array([max_y,max_y]),
                                label="Not incl. in fit",alpha=0.5,color="grey",linewidth=0)

        if ds == datasets[0] or det_type=="sum":
            legend=ax.legend(loc="upper right", ncols=3,frameon=True,facecolor="white")
            ax.set_legend_annotation()
            
        elif ds==datasets[len(datasets)-1]:
            ax.xaxis.set_tick_params(top=False)
            ax.set_legend_logo(position='upper right', logo_type = 'preliminary', scaling_factor=7)
        else:
            ax.xaxis.set_tick_params(top=False)

        if (det_type!="multi"):
            ax.set_legend_logo(position='upper left', logo_type = 'preliminary', scaling_factor=7)




        if (det_type=="multi"):
            ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts / 10 keV")
        ax.set_yscale(scale)
        ax.set_xlim(xlow,xhigh)
        

        ### now plot the residual
        if (det_type!="multi"):
            
            axes_full[1].errorbar(bins,residual,yerr=np.ones(len(residual)),fmt="o",color=vset.blue,markersize=1,linewidth=0.6)
            axes_full[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
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
#plt.show()
plt.savefig("plots/fit_plots/{}_{}_{}_{}_to_{}_{}_{}.pdf".format(name_out,det_type,scale,xlow,xhigh,bin,comp_name))
plt.show()


