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
import pandas as pd

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

### make the name key
names={"Index":[],"Name":[]}
names_U={"Index":[],"Name":[]}
names_Th={"Index":[],"Name":[]}
names_K40={"Index":[],"Name":[]}
names_K42={"Index":[],"Name":[]}

i=0
labels=utils.format_latex(df.keys())

index_U = []
index_Th =[]
index_K40=[]
index_K42=[]

### probably can be done smarter!
for key in df.keys():
    names["Index"].append(i)
    names["Name"].append(labels[i])
    if (key.find("Bi212")!=-1):
        names_U['Name'].append(labels[i])
        names_U["Index"].append(len(index_U))
        index_U.append(i)
    elif(key.find("Bi214")!=-1):
        names_Th['Name'].append(labels[i])
        names_Th["Index"].append(len(index_Th))
        index_Th.append(i)
    elif(key.find("K40")!=-1):
        names_K40['Name'].append(labels[i])
        names_K40["Index"].append(len(index_K40))
        index_K40.append(i)
    elif(key.find("K42")!=-1):
        names_K42['Name'].append(labels[i])
        names_K42["Index"].append(len(index_K42))
        index_K42.append(i)
        
    
    i=i+1
df_key = pd.DataFrame(names)
utils.plot_table(df_key,"plots/key_{}.pdf".format(tree_name))

df_key_U = pd.DataFrame(names_U)
utils.plot_table(df_key_U,"plots/key_U_{}.pdf".format(tree_name))

df_key_Th = pd.DataFrame(names_Th)
utils.plot_table(df_key_Th,"plots/key_Th_{}.pdf".format(tree_name))

df_key_K40 = pd.DataFrame(names_K40)
utils.plot_table(df_key_K40,"plots/key_K40_{}.pdf".format(tree_name))

df_key_K42 = pd.DataFrame(names_K42)
utils.plot_table(df_key_K42,"plots/key_K42_{}.pdf".format(tree_name))


### make the full correlation matrix and subplots
##1) 2vbb rho plot
##2) 212Bi + 208Tl
##3) 214Bi + 214 Bi
##4) 40K
###5) 42K
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

utils.plot_correlation_matrix(matrix,"","plots/Full_matrix_{}.pdf".format(tree_name))

utils.plot_correlation_matrix(matrix[index_U,index_U],"","plots/U_matrix_{}.pdf".format(tree_name))
utils.plot_correlation_matrix(matrix[index_Th,index_Th],"","plots/Th_matrix_{}.pdf".format(tree_name))
utils.plot_correlation_matrix(matrix[index_K40,index_K40],"","plots/K40_matrix_{}.pdf".format(tree_name))
utils.plot_correlation_matrix(matrix[index_K42,index_K42],"","plots/K42_matrix_{}.pdf".format(tree_name))


