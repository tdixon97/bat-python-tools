""" A script to plot the correlation matrix from the hmixfit fit
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
import os
import re
import utils
import json
import mplhep
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

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
parser.add_argument("-c","--cfg",type=str,help="The configuration file for the hmixfit fit")
parser.add_argument("-C","--components",type=str,help="json components config file path",default="components.json")
parser.add_argument("-m","--make_plots",type=bool,help="make scatter plots?",default=True)
parser.add_argument("-N","--step",type=int,help="Number of steps of the markov chain to take ",default=100000)
parser.add_argument("-O","--outdir",type=str,help="output directory to save plots",default="plots/summary/")

args = parser.parse_args()

N=args.step
make_plots=args.make_plots
cfg_path =args.cfg
outdir=args.outdir

with open(cfg_path,"r") as file:
    cfg =json.load(file)

### extract all we need from the cfg dict
fit_name,out_dir,_,_,dataset_name=utils.parse_cfg(cfg)
outfile = out_dir+"/hmixfit-"+fit_name+"/mcmc_small.root"
os.makedirs(outdir+"/"+fit_name,exist_ok=True)

tree_name= "{}_mcmc".format(fit_name)
df =utils.ttree2df(outfile,"mcmc","Phase==1",N=N)

df=df.query("Phase==1")
df=df.drop(columns=['Chain','Iteration','Phase','LogProbability','LogLikelihood','LogPrior','Ar39_homogeneous'])

### make the name key
names={"Index":[],"Name":[]}
names_U={"Index":[],"Name":[]}
names_Th={"Index":[],"Name":[]}
names_K={"Index":[],"Name":[]}

i=0
labels=utils.format_latex(df.keys())

index_U = []
index_Th =[]
index_K=[]

### probably can be done smarter!
for key in df.keys():
    names["Index"].append(i)
    names["Name"].append(labels[i])
    if (key.find("Bi212")!=-1):
        names_Th['Name'].append(labels[i])
        names_Th["Index"].append(len(index_Th))
        index_Th.append(i)
    elif(key.find("Bi214")!=-1):
        names_U['Name'].append(labels[i])
        names_U["Index"].append(len(index_U))
        index_U.append(i)
    elif(key.find("K")!=-1):
        names_K['Name'].append(labels[i])
        names_K["Index"].append(len(index_K))
        index_K.append(i)
      
    i=i+1

### make the full correlation matrix and subplots
matrix = np.zeros(shape=(len(df.keys()),len(df.keys())))

### make the full matrix
labels=utils.format_latex(df.keys())
i=0
j=0
matrix = df.corr()

matrix=np.array(matrix)
with PdfPages('{}/{}/highest_correlations.pdf'.format(outdir,fit_name)) as pdf:


    ### 10 highest overall
    for n in range(20):
        cor,i,j = utils.get_nth_largest(np.abs(matrix),n)
        utils.plot_corr(df,i,j,labels,pdf=pdf)


with PdfPages('{}/{}/correlation_matrix.pdf'.format(outdir,fit_name)) as pdf:

    df_key = pd.DataFrame(names)
    utils.plot_table(df_key,pdf)
    utils.plot_correlation_matrix(matrix,"",pdf,show=False)

    df_key_U = pd.DataFrame(names_U)
    utils.plot_table(df_key_U,pdf)
    utils.plot_correlation_matrix(utils.twoD_slice(matrix,index_U),"",pdf)

    df_key_Th = pd.DataFrame(names_Th)
    utils.plot_table(df_key_Th,pdf)
    utils.plot_correlation_matrix(utils.twoD_slice(matrix,index_Th),"",pdf)

    df_key_K = pd.DataFrame(names_K)
    utils.plot_table(df_key_K,pdf)
    utils.plot_correlation_matrix(utils.twoD_slice(matrix,index_K),"",pdf)


