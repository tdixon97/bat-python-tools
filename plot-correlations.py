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


df=df.drop(columns=['Chain','Iteration','Phase','LogProbability','LogLikelihood','LogPrior','Ar39_homogeneous'])


### make the full correlation matrix
print(df.keys())
matrix = np.zeros(shape=(len(df.keys()),len(df.keys())))

### make the full matrix
labels=utils.format_latex(df.keys())
i=0
j=0
for key1 in df.keys():
    i=0
    for key2 in df.keys():
        if (i>j):
            continue
        x=np.array(df[key1])
        y=np.array(df[key2])
        rangex=(0,max(x))
        rangey=(0,max(y))
        bins=(100,100)
        if (key1!=key2):
            print("{} [1/yr]".format(labels[i]))
            cor=utils.plot_two_dim(x,y,rangex,rangey,"{} [1/yr]".format(labels[i]),
                                                 "{} [1/yr]".format(labels[j]),
                                                 "{} vs {}".format(labels[i],labels[j]),bins)
        else:
            cor=1
        
        matrix[i,j]=cor
        matrix[j,i]=cor
        #print("Correlation {} to {} = {"0}")
        i+=1
    j+=1

utils.plot_correlation_matrix(matrix,"")