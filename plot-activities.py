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
import numpy as np
from IPython.display import display

vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))

parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument("-i","--in_file",type=str,help="file",default="../hmixfit/results/hmixfit-l200a_taup_silver_dataset_m1_norm-analysis.root")
parser.add_argument("-c","--components",type=str,help="json components config file path",default="components.json")
parser.add_argument("-o","--obj",type=str,help="obj to plot 'fit_range' 'bi_range' or 'scaling_factor'",default="fit_range")




datas=["bege","icpc","coax","ppc"]


args = parser.parse_args()
infile=args.in_file
json_file =args.components
obj = args.obj
first_index =utils.find_and_capture(infile,"hmixfit-")
print(infile)
name_out = infile[first_index:-14]


with open(json_file, 'r') as file:
    components = json.load(file, object_pairs_hook=OrderedDict)

dfs={}

for data in datas:
    tree= "counts_l200a_taup_silver_dataset_{}".format(data)

    df =utils.ttree2df(infile,tree)
    df = df[df['fit_range_orig'] != 0]
    dfs[data]=df


if obj=="scaling_factor":
    datas=["bege"]

for data in datas:
    
    fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True, gridspec_kw = {'hspace': 0})

    #axes.set_title("Spectrum breakdown for {}".format(data))
    x=np.array(dfs[data].index)
    y=np.array(dfs[data]["{}_marg_mod".format(obj)])
    y_low = y-np.array(dfs[data]["{}_qt16".format(obj)])
    y_high=np.array(dfs[data]["{}_qt84".format(obj)])-y
    for i in range(len(y_low)):
        if (y_low[i]<0):
            y_low[i]=0
            y_high[i] +=y[i]
            y[i]=0

    ns=0
    upper = np.max(y[ns:]+1.5*y_high[ns:])
    labels=utils.format_latex(np.array(dfs[data]["comp_name"][ns:]))
    print(data)
    print(y_high[0])    
    axes.errorbar(y=labels,x=y[ns:],xerr=[y_low[ns:],y_high[ns:]],fmt="o",color=vset.red,ecolor=vset.blue,markersize=1)
    
    if (obj=="fit_range"):
        axes.set_xlabel("Oberved counts / yr in {} data".format(data))
        axes.set_xlim(0.1,upper)

    elif (obj=="bi_range"):
        axes.set_xlabel("Observed bkg counts / yr in {} data".format(data))
        axes.set_xlim(0.1,upper)
    elif (obj=="scaling_factor"):
        axes.set_xlabel("Scaling factor [1/yr] in {} data".format(data))
        axes.set_xlim(1E-7,upper)
 
    axes.set_yticks(axes.get_yticks())
    axes.set_yticklabels([val for val in labels], fontsize=7)
    plt.xscale("log")
    plt.grid()
    fig.legend()
    #plt.show()
    if (obj!="scaling_factor"):     
        plt.savefig("plots/fit_results_{}_{}_{}.pdf".format(data,obj,name_out))
    else:
       
        plt.savefig("plots/fit_results_{}_{}.pdf".format(obj,name_out))