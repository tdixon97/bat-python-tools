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

t  = 0.36445
gerda_bege =np.array([628, 628-583, 680-628])/32.1
gerda_coax =np.array([698,698-641,747-698])/28.1

M = {"icpc":27.381653933663113,"bege": 6.329419029680365,
     "coax":5.372786451040081,"ppc" :4.89461935216895}

parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument("-i","--in_file",type=str,help="file",default="../hmixfit/results/hmixfit-l200a_taup_silver_dataset_m1_norm-analysis.root")

args=parser.parse_args()
infile=args.in_file

datas=["bege","icpc","ppc","coax"]
idx=0
for data in datas:

    df =pd.DataFrame(utils.ttree2df(infile,data))
    df = df[df['fit_range_orig'] != 0]
 

    if idx==0:
        df_tot = df[df["comp_name"]=="alpha-{}".format(data)]
    else:
        new_row=df[df["comp_name"]== "alpha-{}".format(data)]
        df_tot = pd.concat([df_tot,new_row],ignore_index=True)

               
    idx=idx+1
    
x,y,y_low,y_high=utils.get_from_df(df_tot,"fit_range")
print(df_tot["fit_range_marg_mod"])
print(df_tot["comp_name"])
print(y)
y*=t
y_low*=t
y_high*=t
print(x)
print(y)
print(y_low)
print(y_high)
idx=0

for data in datas:
    a = y[idx]/M[data]
    a_low =y_low[idx]/M[data]
    a_high=y_high[idx]/M[data]
    if (data=="icpc" or data=="ppc"):
        gerda="-"
    elif (data=="bege"):
        gerda="{:.1f}^{{+{:.1f}}}_{{-{:.1f}}}".format(gerda_bege[0],gerda_bege[2],gerda_bege[1])
    else:
        gerda="{:.1f}^{{+{:.1f}}}_{{-{:.1f}}}".format(gerda_coax[0],gerda_coax[2],gerda_coax[1])

    print("{} & {:.1f}^{{+{:.1f}}}_{{-{:.1f}}} & {}\\\\".format(data,a,a_high,a_low,gerda))
    idx+=1

