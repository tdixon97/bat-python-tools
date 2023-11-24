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
parser.add_argument("-m","--make_plots",type=bool,help="make scatter plots?",default=False)

args = parser.parse_args()

outfile=args.out_file
tree_name = args.tree_name
make_plots=args.make_plots


df =utils.ttree2df(outfile,"mcmc")
df=df.query("Phase==1")


df=df.drop(columns=['Chain','Iteration','Phase','LogProbability','LogLikelihood','LogPrior','Ar39_homogeneous'])

print(df)
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
        names_Th["Index"].append(len(index_U))
        index_Th.append(i)
    elif(key.find("Bi214")!=-1):
        names_U['Name'].append(labels[i])
        names_U["Index"].append(len(index_Th))
        index_U.append(i)
    elif(key.find("K")!=-1):
        names_K['Name'].append(labels[i])
        names_K["Index"].append(len(index_K))
        index_K.append(i)
   
        
    
    i=i+1
df_key = pd.DataFrame(names)
utils.plot_table(df_key,"plots/fit_correlations/key_{}.pdf".format(tree_name))

df_key_U = pd.DataFrame(names_U)
utils.plot_table(df_key_U,"plots/fit_correlations/key_U_{}.pdf".format(tree_name))

df_key_Th = pd.DataFrame(names_Th)
utils.plot_table(df_key_Th,"plots/fit_correlations/key_Th_{}.pdf".format(tree_name))

df_key_K = pd.DataFrame(names_K)
utils.plot_table(df_key_K,"plots/fit_correlations/key_K_{}.pdf".format(tree_name))


### make the full correlation matrix and subplots
##1) 2vbb rho plot
##2) 212Bi + 208Tl
##3) 214Bi + 214 Bi
##4) 40K
###5) 42K
matrix = np.zeros(shape=(len(df.keys()),len(df.keys())))

### make the full matrix
labels=utils.format_latex(df.keys())
i=0
j=0
matrix = df.corr()

if (make_plots==True):
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
            
            i+=1
        j+=1
        
matrix=np.array(matrix)
for n in range(10):
    cor,i,j = utils.get_nth_largest(matrix,n)
    
    print("{}: {} to {} = {:0.2f} ".format(n,labels[i],labels[j],cor))
    utils.plot_corr(df,i,j,labels,"plots/fit_correlations/Correlation_{}_to_{}.pdf".format(labels[i],labels[j]))
utils.plot_correlation_matrix(matrix,"","plots/fit_correlations/Full_matrix_{}.pdf".format(tree_name),show=True)
utils.plot_correlation_matrix(utils.twoD_slice(matrix,index_U),"","plots/fit_correlations/Matrix_U_{}.pdf".format(tree_name))
utils.plot_correlation_matrix(utils.twoD_slice(matrix,index_Th),"","plots/fit_correlations/Matrix_Th_{}.pdf".format(tree_name))
utils.plot_correlation_matrix(utils.twoD_slice(matrix,index_K),"","plots/fit_correlations/Matrix_K_{}.pdf".format(tree_name))


