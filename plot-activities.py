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
import pandas as pd
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))

parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument("-i","--in_file",type=str,help="file",default="../hmixfit/results/hmixfit-l200a_taup_silver_dataset_m1_norm-analysis.root")
parser.add_argument("-2","--in_file_2",type=str,help="file",default=None)

parser.add_argument("-c","--components",type=str,help="json components config file path",default="components.json")
parser.add_argument("-o","--obj",type=str,help="obj to plot 'fit_range' 'bi_range' or 'scaling_factor', 'parameter",default="fit_range")
parser.add_argument("-s","--scale",type=float,help="factor to scale the second set of activiies by",default=1)
parser.add_argument("-d","--data",type=str,help="which data type to look at bege icpc ppc coax or all",default="icpc")




### set the parameters
args = parser.parse_args()
infile=args.in_file
infile2=args.in_file_2
scale= args.scale
do_comp=False
datas=[args.data]
if (infile2!=None):
    do_comp=True
json_file =args.components
obj = args.obj
first_index =utils.find_and_capture(infile,"hmixfit-l200a_")

name_out = infile[first_index:-14]
label1=name_out
label2=""
if (do_comp):
    first_index =utils.find_and_capture(infile2,"hmixfit-l200a_")
    label2=infile2[first_index:-14]

label1="Old M2"
label2="New M2"

with open(json_file, 'r') as file:
    components = json.load(file, object_pairs_hook=OrderedDict)

dfs={}
indexs={}
dfs2={}
idx=0


#datas=["icpc","ppc","coax","bege"]
datas=["top","mid_top","mid","mid_bottom","bottom"]
### get the fit results as some dataframes
for data in datas:

    df =pd.DataFrame(utils.ttree2df_all(infile,data))
    df = df[df['fit_range_orig'] != 0]
    dfs[data]=df

    if idx==0:
        df_tot = df
    else:
        new_row=df[df["comp_name"]== "2vbb_{}".format(data)]
        new_row2=df[df["comp_name"]== "K42_hpge_surface_{}".format(data)]

        df_tot = pd.concat([df_tot,new_row,new_row2],ignore_index=True)


    if (do_comp==True):
        df2 =pd.DataFrame(utils.ttree2df_all(infile2,data))
        df2 = df2[df2['fit_range_orig'] != 0]
        dfs2[data]=df2
        
        if idx==0:
            df2_tot = df2
        else:
            new_row=df2[df2["comp_name"]== "2vbb_{}".format(data)]
       
            df2_tot = pd.concat([df2_tot,new_row],ignore_index=True)
    idx=idx+1
print(df_tot)

### if scaling factor or parameter
names=datas
if obj=="scaling_factor" or obj=="parameter":
    datas=["bege"]
    names=["all"]



#### normalise the dataframes
for data,name in zip(datas,names):
    

    if (obj=="parameter"):
        df =df_tot
        df= df.sort_values(by='comp_name')
        
        if (do_comp):
            df2=df2_tot
            df2= df2.sort_values(by='comp_name')
            
        print(df)
      
        x,y,y_low,y_high=utils.get_from_df(df,"fit_range")
        norm = np.array(df["fit_range_orig"])
        x=x/norm
        y=y/norm
        y_low=y_low/norm
        y_high=y_high/norm

        if (do_comp):
            x2,y2,y_low2,y_high2=utils.get_from_df(df2,"fit_range")
            norm2 = np.array(df2["fit_range_orig"])
            x2=x2/norm2
            y2=y2/norm2
            y_low2=y_low2/norm2
            y_high2=y_high2/norm2
            y2*=scale
            y_low2*=scale
            y_high2*=scale
        
    
    else:
        if (obj!="scaling_factor"):
            df=dfs[data]
            if (do_comp):
                df2=dfs2[data]
        else:
            df=df_tot
            if (do_comp):
                df2=df2_tot
        x,y,y_low,y_high=utils.get_from_df(df,obj)
        if (do_comp):
            x2,y2,y_low2,y_high2=utils.get_from_df(df2,obj)
            y2*=scale
            y_low2*=scale
            y_high2*=scale

    ns=0
    labels=utils.format_latex(np.array(df["comp_name"]))
    if (do_comp):
        labels2=utils.format_latex(np.array(df2["comp_name"]))
 
    names=np.array(df["comp_name"])

    indexs=utils.get_index_by_type(names)

    if (do_comp==0):
        utils.make_error_bar_plot(np.arange(len(labels)),labels,y,y_low,y_high,name,name_out,obj)
        utils.make_error_bar_plot(indexs["U"],labels,y,y_low,y_high,name,name_out+"_U",obj,scale="linear",split_priors=True)
        utils.make_error_bar_plot(indexs["Th"],labels,y,y_low,y_high,name,name_out+"_Th",obj,scale="linear",split_priors=True)
        utils.make_error_bar_plot(indexs["K"],labels,y,y_low,y_high,name,name_out+"_K",obj)

    else:
        utils.make_error_bar_plot(np.arange(len(labels)),labels,y,y_low,y_high,name,name_out,obj,y2,y_low2,y_high2,label1,label2,"",1)
        utils.make_error_bar_plot(indexs["U"],labels,y,y_low,y_high,name,name_out,obj,y2,y_low2,y_high2,label1,label2,"U",1,scale="linear",upper=10000,show=True)
        utils.make_error_bar_plot(indexs["2nu"],labels,y,y_low,y_high,name,name_out,obj,y2,y_low2,y_high2,label1,label2,"2nu",1,low=3E5,upper=5E5,scale="linear")

        utils.make_error_bar_plot(indexs["Th"],labels,y,y_low,y_high,name,name_out,obj,y2,y_low2,y_high2,label1,label2,"Th",1,scale="linear",upper=10000,show=True)
        utils.make_error_bar_plot(indexs["K"],labels,y,y_low,y_high,name,name_out,obj,y2,y_low2,y_high2,label1,label2,"K",1,scale="linear")
          
                                  

  