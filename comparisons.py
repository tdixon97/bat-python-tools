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
style_data={"histtype":"fill","alpha":0.3
}
style_mc={}


parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument('-d','--det_type', type=str, help='Detector type',default='all')
parser.add_argument('-l','--lower_x', type=int, help='Low range',default=565)
parser.add_argument('-u','--upper_x', type=int, help='Upper range',default=3000)
parser.add_argument("-s","--scale",type=str,help="Scale",default="log")
parser.add_argument("-1","--in_file_1",type=str,help="file",default="../hmixfit/results/hmixfit-l200a-taup-silver-dataset-raw-histograms.root")
parser.add_argument("-2","--in_file_2",type=str,help="file",default="../hmixfit/results/hmixfit-l200a-taup-silver-dataset-m1-histograms.root")
parser.add_argument("-c","--components",type=str,help="json components config file path",default="components.json")
parser.add_argument("-b","--bins",type=int,help="Binning",default=15)
parser.add_argument("-a","--compare_data",type=int or bool,help="Compare data (true) or model (false)",default=True)

parser.add_argument("-o","--label_1",type=str,help="label for first histo",default="raw")
parser.add_argument("-t","--label_2",type=str,help="label for second hist",default="m1")


### read the args
args = parser.parse_args()
det_type=args.det_type
xlow=args.lower_x
xhigh=args.upper_x
scale=args.scale
in_file_1=args.in_file_1
in_file_2=args.in_file_2
first_index_1 =utils.find_and_capture(in_file_1,"hmixfit-")
name_out_1 = in_file_1[first_index_1:-17]
first_index_2 =utils.find_and_capture(in_file_2,"hmixfit-")
name_out_2 = in_file_2[first_index_2:-17]
json_file =args.components
bin = args.bins
compare_data =args.compare_data
label_1=args.label_1
label_2=args.label_2

def get_hist(obj):
    return obj.to_hist()[132:2982][hist.rebin(bin)]


if det_type=="all" or det_type=="sum":
    datasets = ["l200a_taup_silver_dataset_bege", "l200a_taup_silver_dataset_icpc","l200a_taup_silver_dataset_ppc","l200a_taup_silver_dataset_coax"]
else:
    datasets=["l200a_taup_silver_dataset_{}".format(det_type)]


y=6
if det_type!="all":
    y=4

## create the axes
if (det_type=="all"):
    fig, axes = lps.subplots(len(datasets), 1, figsize=(6, y), sharex=True, gridspec_kw = {'hspace': 0})
else:
    fig, axes_full = lps.subplots(2, 1, figsize=(6, y), sharex=True, gridspec_kw = { 'height_ratios': [8, 2],"hspace":0})


if det_type=="all":
    for ax in axes:
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", direction="in")
else:
    axes=np.array([axes_full[0]])
    
ylowlim=0.01


### open the files the code is pretty complicated could be refactored
with uproot.open(in_file_1) as f1:
    with uproot.open(in_file_2) as f2:
        h1=None
        h2=None
        hdata=None
        for i  in range(len(datasets)):
            ds=datasets[i]
            if (det_type=="all"):
                ax=axes[i]
            else:
                ax=axes[0]
            if compare_data==True:
                name="fitted_data"
            else:
                name="total_model"
         

            if name not in f1[ds]["originals"]:
                raise ValueError("Error: the PDF doesnt exist")
                    
            if name not in f2[ds]["originals"]:
                raise ValueError("Error: the PDF doesnt exist")
            if (det_type=="all" or h1 is None or h2 is None):
                h1 = get_hist(f1[ds]["originals"][name])
                h2 = get_hist(f2[ds]["originals"][name])
            else:
                h1+=get_hist(f1[ds]["originals"][name])
                h2 += get_hist(f2[ds]["originals"][name])
          
            ### compare data
            if (compare_data==False):
                if (det_type=="all" or hdata is None):
                    hdata= get_hist(f1[ds]["originals"]["fitted_data"])
                else:
                    hdata += get_hist(f1[ds]["originals"]["fitted_data"])


                for i in range(hdata.size - 2):
                    if hdata[i] == 0:
                        hdata[i] = 1.05*ylowlim
            

            ### make tje plot
            for i in range(h1.size - 2):
                if h1[i] == 0:
                    h1[i] = 1.05*ylowlim
            
            for i in range(h2.size - 2):
                if h2[i] == 0:
                    h2[i] = 1.05*ylowlim

            if (compare_data==True):
                style_tmp=style_data
            else:
                style_tmp=style_mc

         
            if (det_type=="sum" and ds!=datasets[-1]):
                continue
            ## plot the histos
            if (compare_data==False):
                hdata.plot(ax=ax, **style, **style_data,color=vset.blue,label="Data")

            h1.plot(ax=ax, **style, **style_tmp,color=vset.blue,label=label_1)
            h2.plot(ax=ax, **style, **style_tmp,color=vset.magenta,label=label_2)

            
            data=h1.values()
            bins=h1.axes.centers[0]
            pred=h2.values()

            residual = np.array([m/d if d> -1 else 0 for d, m in zip(data, pred)])

            
            ### now make the plot
       
            if ds == datasets[0] or det_type=="sum":
                ax.legend(loc="upper right", ncols=3)
                ax.set_legend_annotation()
            elif ds==datasets[len(datasets)-1]:
                ax.xaxis.set_tick_params(top=False)
                ax.set_legend_logo(position='upper right', logo_type = 'preliminary', scaling_factor=7)
            else:
                ax.xaxis.set_tick_params(top=False)

            masked_values = np.ma.masked_where((bins > xhigh)  | (bins < xlow), pred)
            masked_values_data = np.ma.masked_where((bins > xhigh)  | (bins < xlow), data)

            maxi = max(masked_values.max(),masked_values_data.max())
            max_y = maxi+2*np.sqrt(maxi)+1

            if (scale=="log"):
                max_y=3*max_y
            ax.set_yscale("linear")
            ax.set_ylim(bottom=ylowlim,top=max_y)      
            if (det_type=="all"):
                ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Counts / 15 keV")
            ax.set_yscale(scale)
            ax.set_xlim(xlow,xhigh)
            
            ### now plot the residual
            if (det_type!="all"):
                
                axes_full[1].plot(bins,residual,color=vset.blue,markersize=0)
                axes_full[1].set_xlabel("Energy (keV)")
                axes_full[1].set_ylabel("Ratio")
                axes_full[1].set_yscale("linear")
                axes_full[1].set_xlim(xlow,xhigh)
                axes_full[1].set_ylim(min(residual),max(residual))
                axes_full[1].xaxis.set_tick_params(top=False)

                plt.tight_layout()

plt.tight_layout()
if (compare_data==True):
    plt.savefig("plots/Comparison_{}_to_{}_{}_data.pdf".format(name_out_1,name_out_2,det_type))
else:
    plt.savefig("plots/Comparison_{}_to_{}_{}_model.pdf".format(name_out_1,name_out_2,det_type))

#plt.show()
