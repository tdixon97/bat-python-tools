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
import mplhep

vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))


style_1d = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}


style_2d = {
    "flow": None,
    "lw": 0.8,
    "cmin":1,
    "cmap":"cividis"

}
parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument("-o","--out_file",type=str,help="file",default="../hmixfit/results/hmixfit-l200a_taup_silver_dataset_m1_norm-_mcmc.root")
parser.add_argument("-c","--components",type=str,help="json components config file path",default="components.json")
parser.add_argument("-t","--tree_name",type=str,help="tree name",default="l200a_taup_silver_dataset_m1_norm")
args = parser.parse_args()

outfile=args.out_file
tree_name = args.tree_name



tree= "{}_mcmc".format(tree_name)
df =utils.ttree2df(outfile,tree)
df=df.query("Phase==1")

print(df)
x=np.array(df["Pb214Bi214_pen_plates"])
y=np.array(df["Pb214Bi214_wls_reflector"])
rangex=(min(x),max(x))
rangey=(min(y),max(y))
bins=(100,100)
utils.plot_two_dim(x,y,rangex,rangey,"$^{214}$Pb+Bi pen plates [decays/yr]","$^{214}$Pb+Bi wls reflector [decays/yr]","",bins)
"""
with uproot.open(outfile) as f:
    
    for key in f.keys():
        histo =f[key].to_hist()
        

        fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True)
        print(key)
        print(key[0]+key[1])
       
        if (key[0]+key[1]=="h1"):
            histo.plot(ax=axes,**style_1d)
        else:
            histo.plot2d(ax=axes,**style_2d)
        substring = key[:-2]
        plt.savefig("plots"+"/"+substring+".pdf")

"""