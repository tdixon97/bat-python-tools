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
parser.add_argument("-2","--in_file_2",type=str,help="file",default=None)

parser.add_argument("-c","--components",type=str,help="json components config file path",default="components.json")
parser.add_argument("-o","--obj",type=str,help="obj to plot 'fit_range' 'bi_range' or 'scaling_factor'",default="fit_range")




datas=["bege","icpc","coax","ppc"]


args = parser.parse_args()
infile=args.in_file
infile2=args.in_file_2
do_comp=False
if (infile2!=None):
    do_comp=True
json_file =args.components
obj = args.obj
first_index =utils.find_and_capture(infile,"hmixfit-l200a_taup_silver_dataset_")

name_out = infile[first_index:-14]
label1=name_out
label2=""
if (do_comp):
    first_index =utils.find_and_capture(infile2,"hmixfit-l200a_taup_silver_dataset_")
    label2=infile2[first_index:-14]


with open(json_file, 'r') as file:
    components = json.load(file, object_pairs_hook=OrderedDict)

dfs={}
indexs={}
dfs2={}

for data in datas:
    tree= "counts_l200a_taup_silver_dataset_{}".format(data)

    df =utils.ttree2df(infile,tree)
    df = df[df['fit_range_orig'] != 0]
    dfs[data]=df
    names=np.array(df["comp_name"])
    indexs[data]=utils.get_index_by_type(names)

    if (do_comp==True):
        df2 =utils.ttree2df(infile2,tree)
        df2 = df2[df2['fit_range_orig'] != 0]
        dfs2[data]=df2

if obj=="scaling_factor":
    datas=["bege"]

for data in datas:
    

    #axes.set_title("Spectrum breakdown for {}".format(data))
    x,y,y_low,y_high=utils.get_from_df(df,obj)
    if (do_comp):
        x2,y2,y_low2,y_high2=utils.get_from_df(df2,obj)

    ns=0
    labels=utils.format_latex(np.array(dfs[data]["comp_name"]))

    if (do_comp==0):
        utils.make_error_bar_plot(np.arange(len(labels)),labels,y,y_low,y_high,data,name_out,obj)
        utils.make_error_bar_plot(indexs[data]["U"],labels,y,y_low,y_high,data,name_out+"_U",obj)
        utils.make_error_bar_plot(indexs[data]["Th"],labels,y,y_low,y_high,data,name_out+"_Th",obj)
        utils.make_error_bar_plot(indexs[data]["K"],labels,y,y_low,y_high,data,name_out+"_K",obj)

    else:
        utils.make_error_bar_plot(np.arange(len(labels)),labels,y,y_low,y_high,data,name_out,obj,y2,y_low2,y_high2,label1,label2,"",1)
        utils.make_error_bar_plot(indexs[data]["U"],labels,y,y_low,y_high,data,name_out,obj,y2,y_low2,y_high2,label1,label2,"U",1)
        utils.make_error_bar_plot(indexs[data]["Th"],labels,y,y_low,y_high,data,name_out,obj,y2,y_low2,y_high2,label1,label2,"Th",1)
        utils.make_error_bar_plot(indexs[data]["K"],labels,y,y_low,y_high,data,name_out,obj,y2,y_low2,y_high2,label1,label2,"K",1)
          
                                  

  