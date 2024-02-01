import re
import pandas as pd
import uproot
import copy
import hist
import os
import sys
import math
from legend_plot_style import LEGENDPlotStyle as lps
from datetime import datetime, timezone
from scipy.stats import poisson
from scipy.stats import norm
lps.use('legend')
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
from hist import Hist
import json
from legendmeta import LegendMetadata
import warnings
from iminuit import Minuit, cost
from scipy.stats import expon
from scipy.stats import truncnorm
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit


def csv_string_to_list(value:str,data_type:type=int)->list:
    """
    Convert a string containg a list of csv values to a list
    Parameters:
        - value (str): The input string
        - data_type(type): The data type to convert to (default: int)
    Returns:
        - a list of type 'data_type'
    Example:
    l = csv_string_to_list("1,2,3",int)

    """
    return [data_type(x) for x in value.split(',')]

def find_and_capture(text:str, pattern:str):
    """Function to  find the start index of a pattern in a string
    Parameters:
        - text: string
        - pattern: string
    Returns:
        -index: Integer corresponding to the first letter of the first instance of the pattern
    """
    match = re.search(pattern, text)
    
    if match:
        start_index = match.end()
        return start_index
    else:
        raise ValueError("pattern {} not found in text {} ".format(pattern,text))
    
def manipulate_title(text:str):
    """Replace all the # with \  and add $ $"""

    
    new_string=""

    for char in text:
        if (char=="#"):
            new_string+="$\\"
        
    return new_string

def format_latex(list_str):
    """Format a string as latex with the radioisotopes formatted"""
    list_new=[]
    for text in list_str:
        modified_string =text.replace("Pb214", "$^{214}$Pb")
        modified_string =modified_string.replace("2vbb", "$2\\nu\\beta\\beta$")
        modified_string=modified_string.replace("Co60","$^{60}$Co")
        modified_string=modified_string.replace("Bi214","$^{214}$Bi")
        modified_string=modified_string.replace("Tl208","$^{208}$Tl")
        modified_string=modified_string.replace("K40","$^{40}$K")
        modified_string=modified_string.replace("K42","$^{42}$K")
        modified_string=modified_string.replace("Bi212","$^{212}$Bi")
        modified_string=modified_string.replace("Ac228","$^{228}$Ac")
        modified_string=modified_string.replace("Ar39","$^{39}$Ar")

        list_new.append(modified_string)
    return list_new
    

def ttree2df_all(filename:str,data:str)->pd.DataFrame:
    """Use uproot to import a TTree and save to a dataframe
    Parameters:
        - filename: file to open
        - tree_name: Which TTree to look at
    Returns:
        Pandas DataFrame of the data
    """

    file = uproot.open(filename)

    # Access the TTree inside the file
    tree =None
    for key in file.keys():
        if (data in key):
            tree = file[key]
    if (tree==None):
        raise ValueError("a tree containing {} not found in the file {}".format(data,filename))
    # Get the list of branches in the TTree
    branches = tree.keys()

    # Define a dictionary to store the branch data
    data = {}

    # Loop through each branch and extract the data
    cache=10000
    leng = tree.num_entries
    idx=0
    list_of_df=[]
    tot_length=0
    for branch_name in branches:
            # Use uproot to read the branch into a NumPy array
            data[branch_name] = tree[branch_name].array()
            df= pd.DataFrame(data)

    return df

def ttree2df(filename:str,data:str,query=None,N=500000)->pd.DataFrame:
    """Use uproot to import a TTree and save to a dataframe
    Parameters:
        - filename: file to open
        - tree_name: Which TTree to look at
    Returns:
        Pandas DataFrame of the data
    """

    file = uproot.open(filename)

    # Access the TTree inside the file
    tree =None
    for key in file.keys():
        if (data in key):
            tree = file[key]
    if (tree==None):
        raise ValueError("a tree containing {} not found in the file {}".format(data,filename))
    # Get the list of branches in the TTree
    branches = tree.keys()

    # Define a dictionary to store the branch data
    data = {}

    # Loop through each branch and extract the data
    cache=10000
    leng = tree.num_entries
    idx=0
    list_of_df=[]
    tot_length=0
    while ((idx+1)*cache<leng and tot_length<N):
        for branch_name in branches:
            # Use uproot to read the branch into a NumPy array
            data[branch_name] = tree[branch_name].array(entry_start=idx*cache,entry_stop=(idx+1)*cache)
        df= pd.DataFrame(data)

        if (query!=None):   
            df=df.query(query)
        idx+=1
        tot_length+=len(df)
        list_of_df.append(df)
    ### read the rest
    for branch_name in branches:
        data[branch_name] = tree[branch_name].array(entry_start=idx*cache,entry_stop = (N-tot_length+idx*cache))
    df= pd.DataFrame(data)

    if (query!=None):   
        df=df.query(query)
        list_of_df.append(df)

    # Create a Pandas DataFrame from the dictionary
    return pd.concat(list_of_df,ignore_index=True)





def plot_two_dim(varx:np.ndarray,vary:np.ndarray,rangex:tuple,rangey:tuple,titlex:str,titley:str,title:str,bins:tuple,show=False,save="",pdf=None):
    """
    A 2D scatter plot
    Parameters:
        - varx: Numpy array of the x data
        - vary: Numpy array of the y data
        - rangex: tuple (low,high) for range of the x axis
        - rangey: tuple (low,high) for range of the y axis
        - titlex: Title for x-axis
        - titley: Title for the y-axis
        - title: Plot title
        - bins: Tuple of (binsx,binsy)
    Returns:
        None
    """

    ## create the axis

    if (show==True):
        fig, axes = lps.subplots(1, 1, figsize=(4,6), sharex=True, gridspec_kw = {'hspace': 0})

        h = axes.hist2d(varx,vary, bins=bins, cmap='viridis', range=[rangex,rangey], cmin=1,edgecolor='none')

        #fig.colorbar(h[3], ax=axes, label='Counts')
        axes.set_xlabel(titlex)
        axes.set_ylabel(titley)
        axes.set_title(title)
        plt.grid()
  

    correlation_coefficient = np.corrcoef(varx, vary)[0, 1]

    # Annotate the plot with correlation coefficient
    if (show==True):
        axes.annotate("Correlation = {:0.2f}".format(correlation_coefficient), (0.6, 0.88), xycoords="axes fraction", fontsize=10)
        if (pdf==None):
            plt.savefig(save)
        else:
            pdf.savefig()
        plt.close()

    return correlation_coefficient

def plot_correlation_matrix(corr_matrix:np.ndarray,title:str,save:str,show=False):
    """
    Plots a correlation matrix 

    Parameters:
    - corr_matrix (numpy.ndarray): 2D NumPy array representing the correlation matrix.
    """

    # Set up the matplotlib figure
    fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True, gridspec_kw = {'hspace': 0})


    # Create a heatmap
    cax = axes.matshow(corr_matrix, cmap='coolwarm',vmin=-1,vmax=1)
    axes.set_title(title)
    
    # Add a colorbar
    cbar = fig.colorbar(cax,shrink=1.0)
    
    cbar.set_label('$\\rho$')
    plt.grid()
    # Show the plot
    plt.savefig(save)
    
    if (show==False):
        plt.close()
    else:
        plt.show()


def plot_table(df,save):
    """ Plot a df as a table"""
    # Create a DataFrame


    # Plot a table
    fig, ax = plt.subplots(figsize=(2, 6*len(df.values)/29))  # Adjust the figsize as needed
    ax.axis('off')  # Turn off axis labels

    # Create the table
    table_data = df.T.reset_index().values
    
    table = ax.table(cellText=df.values,colLabels= df.keys(), cellLoc='center', loc='center',colWidths=(0.2,0.8))

    # Style the table
    table.auto_set_font_size(True)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table.set_zorder(100)
    
    # Show the plot
    plt.savefig(save)
    plt.close()
    

def twoD_slice(matrix,index):
    index=np.array(index)
    matrix_new=matrix[:,index]
    matrix_new = matrix_new[index,:]
    return matrix_new

def get_nth_largest(matrix,n):
    """
    Get the nth largest element in the matrix and its index
    """
    sort_m=matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
                if (i>=j):
                    sort_m[i,j]=0
                
    indices_nth_largest = np.argsort(sort_m.flatten())[-(n+1)]
    
    row_indices, col_indices = np.unravel_index(indices_nth_largest, matrix.shape)  
    
    return sort_m[row_indices,col_indices],row_indices,col_indices

def plot_corr(df,i,j,labels,save,pdf=None):
    key1=df.keys()[i]
    key2=df.keys()[j]
    x=np.array(df[key1])
    y=np.array(df[key2])
    rangex=(0,max(x))
    rangey=(0,max(y))
    bins=(100,100)
        
    cor=plot_two_dim(x,y,rangex,rangey,"{} [1/yr]".format(labels[i]),
                                                    "{} [1/yr]".format(labels[j]),
                                                    "{} vs {}".format(labels[i],labels[j]),bins,True,save,pdf=pdf)
    


def make_error_bar_plot(indexs,labels:list,y:np.ndarray,ylow:np.ndarray,yhigh:np.ndarray,data="all",name_out=None,obj="parameter",
                        y2=None,ylow2=None,yhigh2=None,label1=None,label2=None,extra=None,do_comp=0,low=0.1,scale="log",upper=0,
                        save_path ="plots/fit_results/fit_results",show=False,pdf=None,data_band =None,categories=None,split_priors=False
                        ):
    """
    Make the error bar plot
    """
    indexs=np.array(indexs,dtype="int")
    vset = tc.tol_cset('vibrant')
    labels=np.array(labels)
    y=y[indexs]
    labels=labels[indexs]
    print(len(y),len(labels))
    ylow=ylow[indexs]
    yhigh=yhigh[indexs]

    if (do_comp==True):
        
        ylow2=ylow2[indexs]
        yhigh2=yhigh2[indexs]
        y2=y2[indexs]

    height= 2+4*len(y)/29
    fig, axes = lps.subplots(1, 1, figsize=(6,2.5), sharex=True, gridspec_kw = {'hspace': 0})

    ### get the indexs with prior and those without priors
    ### ------------------------------------------------------------
    xin=np.arange(len(labels))
    index_prior=[]
    index_no_prior=[]
 

    for i in range(len(labels)):
        label = labels[i]
        if ("cables" in label) or ("pen" in label) or ("sipm" in label) or ("insulator" in label):
            index_prior.append(True)
            index_no_prior.append(False)
        else:
            index_prior.append(False)
            index_no_prior.append(True)


    ### split into contributions with and without priors
    if (split_priors):
        index_prior=np.array(index_prior)
        index_no_prior=np.array(index_no_prior)
        yo=y
        ylowo=ylow
        yhigho=yhigh

        y=yo[index_prior]
        ylow=ylowo[index_prior]
        yhigh=yhigho[index_prior]
        y2=yo[index_no_prior]
        ylow2=ylowo[index_no_prior]
        yhigh2=yhigho[index_no_prior]
        xin1=xin[index_prior]
        xin2=xin[index_no_prior]
  
    ### either split by category or not
    if categories is None:
        if (do_comp==False and split_priors==False):
            axes.errorbar(y=xin+0.15*len(y)/30,x=y,xerr=[abs(ylow),abs(yhigh)],fmt="o",color=vset.blue,ecolor=vset.cyan,markersize=1,label="MC")
        else:
            if (split_priors):
                axes.errorbar(y=xin1,x=y,xerr=[abs(ylow),yhigh],fmt="o",color=vset.blue,ecolor=vset.cyan,markersize=1,
                            label="With prior")
                axes.errorbar(y=xin2,x=y2,xerr=[abs(ylow2),yhigh2],fmt="o",color=vset.red,ecolor=vset.orange,markersize=1,
                            label="Without prior")
            else:
                axes.errorbar(y=xin+0.15*len(y)/30,x=y,xerr=[ylow,yhigh],fmt="o",color=vset.blue,ecolor=vset.cyan,markersize=1,label=label1)
                axes.errorbar(y=xin-0.15*len(y)/30,x=y2,xerr=[ylow2,yhigh2],fmt="o",color=vset.orange,ecolor=vset.magenta,markersize=1,label=label2)
                
    else:
        if (do_comp==True):
            raise ValueError("Splitting by category not implemented for comparison")
        
        cat_sort=np.argsort(categories)
       
        labels=labels[cat_sort]
        categories=categories[cat_sort]
        y=y[cat_sort]
        ylow=ylow[cat_sort]
        yhigh=yhigh[cat_sort]
        colors=[vset.blue,vset.teal,vset.magenta,vset.cyan]
        for cat,col in zip(sorted(set(categories)),colors):
            y_tmp = y[categories==cat]
            ylow_tmp = ylow[categories==cat]
            yhigh_tmp = yhigh[categories==cat]
            xin_tmp = xin[categories==cat]
            axes.errorbar(y=xin_tmp,x=y_tmp,xerr=[ylow_tmp,yhigh_tmp],fmt="o",color=col,ecolor=col,markersize=1,label=cat)


    ### add a band of the data           
    if data_band is not None:
        axes.axvline(x=data_band[0], color='red', linestyle='--', label='Data')

        axes.axvspan(xmin=data_band[0]-data_band[1],xmax=data_band[0]+data_band[2], color=vset.orange, alpha=0.3)

    if (upper==0):
        upper = np.max(y+1.5*yhigh)
        if (do_comp==True or split_priors==True):
            upper=max(upper,np.max(y2+1.5*yhigh2))
            
    axes.set_yticks(xin, labels)

    if (data_band is not None):
        upper =data_band[0]+7*data_band[2]
    if (obj=="fit_range"):
        axes.set_xlabel("Oberved counts / yr in {} data".format(data))
        axes.set_xlim(0.1,upper)

    elif (obj=="bi_range"):
        axes.set_xlabel("Observed bkg counts / yr in {} data".format(data))
        axes.set_xlim(low,upper)
    elif (obj=="scaling_factor"):
        axes.set_xlabel("Scaling factor [1/yr] ")
        axes.set_xlim(1E-7,upper)
    elif obj=="parameter":
        axes.set_xlabel("Decays / yr [1/yr]")
        axes.set_xlim(low,upper)
 
    axes.set_yticks(axes.get_yticks())
    axes.set_yticklabels([val for val in labels], fontsize=11)
    axes.set_xscale(scale)

    plt.grid()
    if (do_comp==True or split_priors==True or data_band is not None or categories is not None):
        leg=axes.legend(loc='best',edgecolor="black",frameon=True, facecolor='white',framealpha=1)
        leg.set_zorder(10)    

    #plt.show()
    if (do_comp==True):
        name_out="comp_{}_to_{}_{}".format(label1,label2,extra)
   

    if (pdf is None):
        if (obj!="scaling_factor"):     
            plt.savefig("{}_{}_{}_{}.pdf".format(save_path,data,obj,name_out))
        else:
            plt.savefig("{}_{}_{}.pdf".format(save_path,obj,name_out))
    else:
        pdf.savefig()

    if (show==True):
        plt.show()
    else:
        plt.close()

def replace_e_notation(my_string):
    # Using regular expression to match e+0n where n is any character
    pattern = re.compile(r'e\+0(.)')
    modified_string = re.sub(pattern, r'\\cdot 10^{\1}', my_string)
    return modified_string


def priors2table(priors:dict):
    """ Convert the priors json into a latex table"""
    print(json.dumps(priors,indent=1))

    convert_fact=1/31.5
    print("\multicolumn{1}{c}{\cellcolor[HTML]{CBCEFB}\\textbf{Source} &\multicolumn{1}{c}{\cellcolor[HTML]{CBCEFB} \\textbf{} Decay} & \multicolumn{1}{c}{\cellcolor[HTML]{CBCEFB} \\textbf{Activity [$\mu$Bq]} & \multicolumn{1}{c}{\cellcolor[HTML]{CBCEFB}\\textbf{Type} \\\\ \hline \hline ")
    first_Bi=0
    first_Tl=0
    for comp in priors["components"]:

        source = comp["full-name"]
        if ("Bi212Tl208" in comp["name"] and first_Tl==0):
            decay = "$^{212}$Bi+$^{208}$Tl"
            first_Tl=1
        elif ("Pb214Bi214" in comp["name"] and first_Bi==0):
            decay = "$^{214}$Pb+$^{214}$Bi"
            first_Bi=1
        else:
            decay= ""
        type_meas = comp["type"]
        if (type_meas=="icpms"):
            type_meas="ICP-MS"
        if (type_meas=="guess"):
            type_meas="Guess"
        if (type_meas=="hpge"):
            type_meas="HPGe"
        prior = comp["prior"]
        if ("gaus" in prior):
            split_values = prior.split(":")

            param1, param2, param3 = map(float, split_values[1].split(","))
            a=-param2/param3

            rv = truncnorm(a=a,b=5, loc=param2, scale=param3)
            high=param2+5*param3
            low_err = 0 if param3 > param2 else param2-param3
            high_err= param3+param2
            best=param2
            low_err*=convert_fact
            high_err=convert_fact*param3
            best*=convert_fact
            meas = "${:.2g} \pm{:.2g}$".format(best,high_err)
            meas = replace_e_notation(meas)
        elif ("exp" in prior):
            split_parts = prior.split("/")

            # Extract the upper limit from the exponenital
            if len(split_parts) > 1:
                upper_limit = float( split_parts[1][:-1])
                rv= expon(scale=upper_limit/2.3)
                high = 2*upper_limit
                low_err=0
                high_err= upper_limit
                high_err*=convert_fact
                meas="$<{:.2g}$".format(high_err)
                meas = replace_e_notation(meas)

        print("{} & {} & {} & {} \\\\".format(source,decay,meas,type_meas))

def get_index_by_type(names):
    """ Get the index for each type (U,Th,K)"""
    i=0
    index_U = []
    index_Th =[]
    index_K=[]
    index_2nu=[]
    
    for key in names:
       
        if (key.find("Bi212")!=-1):
            index_Th.append(i)
        elif(key.find("Bi214")!=-1):
            index_U.append(i)
        elif(key.find("K42")!=-1):
        
            index_K.append(i)    
        elif (key.find("2v")!=-1):
            index_2nu.append(i)
        i=i+1

    return {"U":index_U,"2nu":index_2nu,"Th":index_Th,"K":index_K}


def get_from_df(df,obj):
    """Get the index, y and errors from dataframe"""
    x=np.array(df.index)
    y=np.array(df["{}_marg_mod".format(obj)])
    y_low = y-np.array(df["{}_qt16".format(obj)])
    y_high=np.array(df["{}_qt84".format(obj)])-y
    for i in range(len(y_low)):
        if (y_low[i]<0):
            y_low[i]=0
            y_high[i] +=y[i]
            y[i]=0

    return x,y,y_low,y_high


def integrate_hist(hist,low,high):
    """ Integrate the histogram"""

    bin_centers= hist.axes.centers[0]

    values = hist.values()
    lower_index = np.searchsorted(bin_centers, low, side="right")
    upper_index = np.searchsorted(bin_centers, high, side="left")
    bin_contents_range =values[lower_index:upper_index]
    bin_centers_range=bin_centers[lower_index:upper_index]

    return np.sum(bin_contents_range)

def get_total_efficiency(det_types,cfg,spectrum,regions,pdf_path,det_sel="all",mc_list=None):
    eff_total={}
    ### creat the efficiency maps (per dataset)
    for det_name, det_info in det_types.items():
        
        det_list=det_info["names"]
        effs={}
        for key in regions:
            effs[key]={}

        for det,named in zip(det_list,det_info["types"]):
            eff_new,good = get_efficiencies(cfg,spectrum,det,regions,pdf_path,named,"mul_surv",mc_list=mc_list)
            if (good==1 and (named==det_sel or det_sel=="all")):
                effs=sum_effs(effs,eff_new)

        eff_total[det_name]=effs

    return eff_total


def get_efficiencies(cfg,spectrum,det_type,regions,pdf_path,name,spectrum_fit="",mc_list=None,type_fit="icpc"):
    """ Get the efficiencies"""

    if (det_type in ["icpc","ppc","coax","bege"]):
        type_fit=det_type

    effs={}
    for key in regions:
        effs[key]={}
  
    if (spectrum_fit==""):
        spectrum_fit=spectrum

    if mc_list is None:
        for key,region in regions.items():
            effs[key]["2vbb_bege"]=0
            effs[key]["2vbb_coax"]=0
            effs[key]["2vbb_ppc"]=0
            effs[key]["2vbb_icpc"]=0
            effs[key]["K42_hpge_surface_bege"]=0
            effs[key]["K42_hpge_surface_coax"]=0
            effs[key]["K42_hpge_surface_ppc"]=0
            effs[key]["K42_hpge_surface_icpc"]=0
            effs[key]["alpha_ppc"]=0
            effs[key]["alpha_bege"]=0
            effs[key]["alpha_coax"]=0
            effs[key]["alpha_icpc"]=0
        for key in cfg["fit"]["theoretical-expectations"].keys():
            if ".root" in key:
                filename = key
        if "{}/{}".format(spectrum_fit,type_fit) in cfg["fit"]["theoretical-expectations"][filename]:
        
            icpc_comp_list=cfg["fit"]["theoretical-expectations"][filename]["{}/{}".format(spectrum_fit,type_fit)]["components"]
        else:
            warnings.warn("{}/{} not in MC PDFs".format(spectrum_fit,det_type))
            return effs,0
    else:

        icpc_comp_list=mc_list

    comp_list = copy.deepcopy(icpc_comp_list)
    for comp in comp_list:
        
        for key in comp["components"].keys():
            par = key
        ## now open the file
      
        if "root-file" in comp.keys():
            file = uproot.open(pdf_path+comp["root-file"])
            
            if "{}/{}".format(spectrum,det_type) in file:
                hist = file["{}/{}".format(spectrum,det_type)]
                N  = int(file["number_of_primaries"])
                hist = hist.to_hist()
                for key,region in regions.items():
                    eff=0
                    for region_tmp in region:
                        eff+=float(integrate_hist(hist,region_tmp[0],region_tmp[1]))
                    effs[key][par]=eff/N
            else:

                warnings.warn("{}/{} not in MC PDFs".format(spectrum,det_type))
                for key,region in regions.items():
                    effs[key][par]=0

        ### a formula not a file
        else:
            comps = comp["components"]

            for name in comps:
                formula = comps[name]["TFormula"]

            ## curren   y only pol1 implemented
            if "pol1" not in formula:
                raise ValueError("Currently integration only works for pol1")
        
            split_values = formula.split(":")
            p0, p1 = map(float, split_values[1].split(","))
            for key,region in regions.items():
                eff =0
                for region_tmp in region:

                    eff+= p0*(region_tmp[1]-region_tmp[0])+p1*(region_tmp[1]*region_tmp[1]-region_tmp[0]*region_tmp[0])/2
                effs[key][par]=eff

                if (det_type not in ["icpc","bege","ppc","coax"]):
                    effs[key][par]=0

    return effs,1
   

def sum_effs(eff1,eff2):
    """ Sum the two efficiency dictonaries"""


    dict_sum={}

    for key in set(eff1) | set(eff2):  # Union of keys from both dictionaries

        ## sum two layers
        dict_sum[key]={}
     
        if (isinstance(eff1[key],dict) or isinstance(eff2[key],dict)):
      
            for key2 in set(eff1[key]) | set(eff2[key]):
                dict_sum[key][key2] = eff1[key].get(key2, 0) + eff2[key].get(key2, 0)

        ## sum one layer
        else:
            dict_sum[key]= eff1.get(key, 0) + eff2.get(key, 0)
    return dict_sum

def get_data_counts_total(spectrum,det_types,regions,file,det_sel="all",
                          key_list=["full","2nu","Tlcompton","Tlpeak","K40","K42"]):

    data_counts_total ={}
    for det_name,det_info in det_types.items():

        det_list=det_info["names"]
        dt=det_info["types"]
        data_counts={}
        for key in key_list:
            data_counts[key]=0 
        for det,type in zip(det_list,dt):
            if (type==det_sel or det_sel=="all"):
                data_counts = sum_effs(data_counts,get_data_counts(spectrum,det,regions,file))

        data_counts_total[det_name]=data_counts

    return data_counts_total


def create_efficiency_likelihood(data_surv,data_cut):
    """Create the likelihood function"""

    def likelihood(Ns,Nb,eff_s,eff_b):
        """The likelihood function"""

        logL=0
        preds_surv =np.array([Nb*eff_b,Nb*eff_b+Ns*eff_s,Nb*eff_b])
        preds_cut =np.array([Nb*(1-eff_b),Nb*(1-eff_b)+Ns*(1-eff_s),Nb*(1-eff_b)])

        logL+=sum(poisson.logpmf(data_surv, preds_surv))
        logL+=sum(poisson.logpmf(data_cut, preds_cut))
       
        return -logL
    return likelihood


def create_graph_likelihood(func,x,y,el,eh):

    def likelihood(*pars):
        logL=0

        logLs=(func(x,*pars)>y)*(- np.power((func(x,*pars) -y),2) / (2*eh))
        logLs+=(func(x,*pars)<=y)*(- np.power((func(x,*pars) -y),2) / (2*el))
        logL = sum(logLs)

        return -logL

    return likelihood


def create_counting_likelihood(data,bins):
    """Create the likelihood function"""

    def likelihood(Ns,Nb):
        """The likelihood function"""

        logL=0
        preds_surv =np.array([Nb*bins[0]/bins[1],Nb+Ns,Nb*bins[2]/bins[1]])
      
        logL+=sum(poisson.logpmf(data, preds_surv))

        return -logL
    return likelihood


def plot_counting_calc(energies,data,values):
    """ Make a plot to show the efficiency calculation"""
    
    vset = tc.tol_cset('vibrant')
    Nb=values["Nb"]
    Ns=values["Ns"]
    
    bkg= np.array([Nb,Nb,Nb,Nb])
    sig= np.array([0,Ns,0])
   
   
    
    fig, axes = lps.subplots(1, 1, figsize=(4,4), sharex=True, gridspec_kw = {'hspace': 0})
    centers=[(energies[i] + energies[i + 1]) / 2 for i in range(len(energies) - 1)]
  
    axes.bar(centers, data, width=np.diff(energies), edgecolor=vset.blue, align='center',color=vset.blue,alpha=0.3,linewidth=0,label="Data")
    axes.step(energies,bkg, where="mid",color=vset.red,alpha=1,linewidth=1,label="Bkg.")
    axes.step(centers,sig, where="mid",color=vset.teal,alpha=1,linewidth=1,label="Sig")

    axes.legend(loc="best")
   
    axes.set_xlim(energies[0],energies[-1])

    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("Counts/bin")

   
    axes.set_ylabel("Counts/bin")

   
  
    axes.set_title("Events for {} keV".format(centers[1]),fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/relative_intensity/counts_calc_{}.pdf".format(centers[1]))
    plt.close()

def plot_relative_intensities_triptych(outputs,data,data_err,orders,title,savepath,no_labels=False):
    ### now loop over the keys
    
    vset = tc.tol_cset('vibrant')

    ratios=[]
    
    names=[]
    for i in range(len(outputs)):
        ratios_tmps =[]
        for component in orders:
            ratio_tmp = outputs[i][component]
            ratios_tmps.append(ratio_tmp)
            
            if (i==0):
                if ((component!="front_end_electronics" )and (component!="hpge_support_copper")):
                    names.append(component)
                elif component=="front_end_electronics":
                    names.append("fe_electronics")
                else:
                    names.append("hpge_copper")
        ratios_tmps=np.array(ratios_tmps)
        ratios.append(ratios_tmps)
    names=np.array(names)
 
    fig, axes = lps.subplots(1, 3,figsize=(6, 3.5), sharex=False, sharey=True,gridspec_kw = { "hspace":0})

    for i in range(len(outputs)):
        axes[i].set_xlim(0,1.2*max(max(ratios[i]),data[i]+data_err[i]))
        axes[i].set_title(title[i],fontsize=16)
        axes[i].errorbar(ratios[i],names,color=vset.blue,fmt="o",label="MC")
        if (i==1):
            axes[i].set_xlabel("Relative Intensity [%]")
        axes[i].axvline(x=data[i],color=vset.red,label="data")
        axes[i].axvspan(xmin=data[i]-data_err[i], xmax=data[i]+data_err[i], alpha=0.2, color=vset.orange)
        if (i==0):
            axes[i].set_yticks(np.arange(len(names)),names,rotation=0,fontsize=11)
        if (i==2):
            axes[i].legend(loc="best",edgecolor="black",frameon=True, facecolor='white',framealpha=1)
        
    plt.tight_layout()
    plt.savefig(savepath)

def plot_relative_intensities(outputs,data,data_err,orders,title,savepath,no_labels=False):
    ### now loop over the keys
    
    vset = tc.tol_cset('vibrant')

    ratios=[]

    names=[]
    for component in orders:
        ratio_tmp = outputs[component]
        ratios.append(ratio_tmp)
        names.append(component)
     
    ratios=np.array(ratios)
    names=np.array(names)
    print(names)

    fig, axes = lps.subplots(1, 1,figsize=(3, 3), sharex=True, gridspec_kw = { "hspace":0})
    axes.set_xlim(0,10+max(max(ratios),data+data_err))
    axes.set_title(title,fontsize=14)
    axes.errorbar(ratios,names,color=vset.blue,fmt="o",label="MC")
    axes.set_xlabel("Relative Intensity [%]")
    axes.set_yticks(np.arange(len(names)),names,rotation=0,fontsize=7)
    axes.axvline(x=data,color=vset.red,label="data")
    axes.axvspan(xmin=data-data_err, xmax=data+data_err, alpha=0.2, color=vset.orange)
    axes.legend(loc="best",edgecolor="black",frameon=True, facecolor='white',framealpha=1)
    plt.tight_layout()
    plt.savefig(savepath)



def normalized_poisson_residual(mu, obs):
    """" code to compute normalised poisson residuals"""
    
    
    if (obs<1):
        obs=0
    if mu < 50:
        ## get mode

        mode =math.floor(mu)

        if (obs==0 and mode==0):
            return 0
        elif obs<mode:
            prob = poisson.cdf(obs, mu, loc=0)
            sign=-1
        else:
            prob = 1-poisson.cdf(obs-1,mu,loc=0)
            sign=1
        return sign*norm.ppf(1-prob)
    else:
        return (obs - mu) / np.sqrt(mu)


def plot_eff_calc(energies,data_surv,data_cut,values):
    """ Make a plot to show the efficiency calculation"""
    
    vset = tc.tol_cset('vibrant')
    Nb=values["Nb"]
    Ns=values["Ns"]
    eff_s=values["eff_s"]
    eff_b =values["eff_b"]
    bkg= np.array([Nb*eff_b,Nb*eff_b,Nb*eff_b,Nb*eff_b])
    sig= np.array([0,Ns*eff_s,0])
   
    bkg_cut= np.array([Nb*(1-eff_b),Nb*(1-eff_b),Nb*(1-eff_b),Nb*(1-eff_b)])
    sig_cut= np.array([0,Ns*(1-eff_s),0])
    
    fig, axes = lps.subplots(2, 1, figsize=(4,4), sharex=True, gridspec_kw = {'hspace': 0})
    centers=[(energies[i] + energies[i + 1]) / 2 for i in range(len(energies) - 1)]
  
    axes[0].bar(centers, data_surv, width=np.diff(energies), edgecolor=vset.blue, align='center',color=vset.blue,alpha=0.3,linewidth=0,label="Data")
    axes[0].step(energies,bkg, where="mid",color=vset.red,alpha=1,linewidth=1,label="Bkg.")
    axes[0].step(centers,sig, where="mid",color=vset.teal,alpha=1,linewidth=1,label="Sig")

    axes[0].legend(loc="best")
    axes[1].bar(centers, data_cut, width=np.diff(energies), edgecolor=vset.blue, align='center',color=vset.blue,alpha=0.3,linewidth=0)
    axes[1].step(energies,bkg_cut, where="mid",color=vset.red,alpha=1,linewidth=1)
    axes[1].step(centers,sig_cut, where="mid",color=vset.teal,alpha=1,linewidth=1)
   
    axes[1].set_xlim(energies[0],energies[-1])
    axes[0].set_xlim(energies[0],energies[-1])

    axes[0].set_xlabel("Energy [keV]")
    axes[0].set_ylabel("Counts/bin")

    axes[1].set_xlabel("Energy [keV]")

    axes[1].set_ylabel("Counts/bin")
    axes[0].set_ylabel("Counts/bin")

    axes[1].xaxis.set_tick_params(top=False)
  
    axes[0].set_title("Events passing LAr (top) and cut (bottom) for {} keV".format(centers[1]),fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/lar/eff_calc_{}.pdf".format(centers[1]))


def extract_prior(prior:str):
    """
    Extracts the prior as a sccipy object (only works for Gauss or Expon)
    
    """

    #gaussian prior
    rv=None
    if ("gaus" in prior):
        split_values = prior.split(":")

        param1, param2, param3 = map(float, split_values[1].split(","))
        a=-param2/param3

        rv = truncnorm(a=a,b=5, loc=param2, scale=param3)
        high=param2+5*param3
        low_err = 0 if param3 > param2 else param2-param3
        high_err= param3+param2

    elif ("exp" in prior):
        split_parts = prior.split("/")

        # Extract the upper limit from the exponenital
        if len(split_parts) > 1:
            upper_limit = float( split_parts[1][:-1])
            rv= expon(scale=upper_limit/2.3)
            high = 2*upper_limit
            low_err=0
            high_err= upper_limit
        else:
            raise ValueError("exp prior doesnt contain the part /UpperLimit) needed")
    else:
        raise ValueError("Only supported priors are gaus and exp currently")
    
    return rv,high,(low_err,high_err)


def parse_cfg(cfg_dict:dict)->(str,str,str,list,list,str):
    """
    Extract a bunch of information from the hmixfit cfg files
    Parameters:
        - cfg_dict: dictonary (cfg file for hmixfit)
    Returns
        - the fit name
        - the output directory of fit results
        - the list of detector types, 
        - the ranges for each fit
        - the name of the dataset
    """

    fit_name=cfg_dict["id"]
    out_dir="../hmixfit/"+cfg_dict["output-dir"]
    det_types=[]
    ranges=[]
    os.makedirs("plots/summary/{}".format(fit_name),exist_ok=True)

    for key in cfg_dict["fit"]["theoretical-expectations"]:
        for spec in cfg_dict["fit"]["theoretical-expectations"][key]:
            det_types.append(spec)
            ranges.append(cfg_dict["fit"]["theoretical-expectations"][key][spec]["fit-range"])
        dataset_name =key[:-5]

    dataset_name=dataset_name.replace("-","_").replace(".","_")

    return fit_name,out_dir,det_types,ranges,dataset_name

def plot_mc_no_save(axes,pdf,name="",linewidth=0.6,range=(0,4000)):
    
    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("counts/10 keV")

    axes.set_yscale("linear")
    axes.set_xlim(range)
    pdf.plot(ax=axes, linewidth=linewidth, yerr=False,flow= None,histtype="step",label=name,color="blue")
    #pdf.plot(ax=axes, linewidth=0.8, yerr=False,flow= None,histtype="fill",alpha=0.15)

    return axes

def compare_mc(max_energy,pdfs,decay,order,norm_peak,pdf,xlow,xhigh,scale,linewidth=0.6):
    """Compare MC files"""
    fig, axes = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})
    max_c=0
    axes.set_xlim(xlow,xhigh)

    for name in order:
 
        Peak_counts =pdfs[decay+"_"+name][norm_peak]

        pdf_norm = scale_hist(pdfs[decay+"_"+name],1/Peak_counts)

        if (pdf_norm[max_energy]>max_c):
            max_c= pdf_norm[max_energy]
 
        axes=plot_mc_no_save(axes,pdf_norm,name=name,linewidth=linewidth)
        
    axes.legend(loc='best',edgecolor="black",frameon=True, facecolor='white',framealpha=1,fontsize=8)    
    axes.set_xlim(xlow,xhigh)
    axes.set_yscale(scale)
    if (scale!="log"):
        axes.set_ylim(0,1.1*max_c)
 
    
    pdf.savefig()

def compare_intensity_curves(intensity_map,order,save=""):
    """Compare intensity curves"""

    fig, axes = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})
   
    maxI=0
    for name in order:
        map_tmp =intensity_map[name]
        E=[]
        I =[]
        for peak in map_tmp:
            E.append(peak)
            I.append(map_tmp[peak])

        E=np.array(E)
        I=np.array(I)*100

        axes.plot(E,I,label=name,linewidth=0.6)
        axes.set_xlim(min(E)-100,max(E)+100)
        if (max(I)>maxI):
            maxI = max(I)

        axes.set_ylim(0,maxI+30)
    
        axes.set_xlabel("Peak Energy [keV]")
        axes.set_ylabel("Relative Intensitiy [%]")
    axes.legend(loc='upper right',edgecolor="black",frameon=True, facecolor='white',framealpha=1,fontsize=8)    

    plt.savefig(save)


def plot_N_Mc(pdfs,labels,name,save_pdf,data,range_x,range_y,scale,colors):
    """Plot the MC files"""
    
    vset = tc.tol_cset('vibrant')

    fig, axes = lps.subplots(1, 1,figsize=(6, 4), sharex=True, gridspec_kw = { "hspace":0})
    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("counts/10 keV")
    axes.set_title(name)
    axes.set_yscale(scale)
    axes.set_xlim(range_x)
    axes.set_ylim(range_y)
    if (data is not None):
        data.plot(ax=axes,yerr=False,flow=None,histtype="fill",alpha=0.3,label="Data",color=vset.blue)
    for pdf,c,l in zip(pdfs,colors,labels):
        pdf.plot(ax=axes, linewidth=.6,color=c,yerr=False,flow= None,histtype="step",label=l)
   

    axes.legend(loc='best',edgecolor="black",frameon=True, facecolor='white',framealpha=1)    

    save_pdf.savefig()


def plot_mc(pdf,name,save_pdf,data=None,range_x=(500,4000),range_y=(0.01,5000),pdf2=None,scale="log",label=""):
    """Plot the MC files"""
    vset = tc.tol_cset('vibrant')

    fig, axes = lps.subplots(1, 1,figsize=(5, 4), sharex=True, gridspec_kw = { "hspace":0})
    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("counts/10 keV")
    axes.set_title(name)
    axes.set_yscale(scale)
    axes.set_xlim(range_x)
    axes.set_ylim(range_y)
    if (data is not None):
        data.plot(ax=axes,yerr=False,flow=None,histtype="fill",alpha=0.3,label="Data",color=vset.blue)
    pdf.plot(ax=axes, linewidth=1,color=vset.red, yerr=False,flow= None,histtype="step",label=label)
    if (pdf2 is not None):
        pdf2.plot(ax=axes, linewidth=1,color=vset.red,alpha=0.6, yerr=False,flow= None,histtype="step",label="90 pct upper limit")

    if (label!=""):
        axes.legend(loc='best',edgecolor="black",frameon=True, facecolor='white',framealpha=1)    

    save_pdf.savefig()

    plt.close()

def vals2hist(vals,hist):
    for i in range(hist.size - 2):
        hist[i]=vals[i]
    return hist

def slow_convolve(priors,pdfs,rvs,n_samp=10000):
    """
    Convolution done in a slow way but fast enough
    """
    ### get the number of bins and components

    n_bins=0
    n_comp=0
    for comp in priors["components"]:
    
        vals=np.array(pdfs[comp["name"]].values())
        n_bins=len(vals)
        n_comp+=1


    output = np.zeros((n_bins,n_samp))

    for i in range(n_samp):
   
        h_tot = np.zeros(n_bins)
        for comp in priors["components"]:
            rv=rvs[comp["name"]]

            a = rv.rvs(1)
           
            a=np.array(a)
            h_tot+=np.array(pdfs[comp["name"]].values())*a[0]

        output[:,i]=h_tot
      

    return output



    

def plot_pdf(rv,high,samples=np.array([]),pdf_obj=None,name=""):
    """ Plot the PDF and optionally some samples"""
    vset = tc.tol_cset('vibrant')

    fig, axes = lps.subplots(1, 1,figsize=(5, 4), sharex=True, gridspec_kw = { "hspace":0})

    x=np.linspace(0,high,10000)

    pdf = rv.pdf(x)
    axes.plot(x,pdf,color=vset.red,label="prior distribution")

    if (len(samples)>0):
        axes.hist(samples,range=(0,high),bins=100,label="Samples",density=True,color=vset.blue,alpha=0.4)
    axes.set_xlabel("decays/yr")
    axes.legend(loc="best")
    axes.set_ylabel("Probability Density ")
    axes.set_xlim(0,high)
    axes.set_title(name)
    if (pdf_obj is not None):
        pdf_obj.savefig()

    plt.close()

def scale_hist(hist,scale):
    """Scale a hist object"""
    hist_scale=copy.deepcopy(hist)
    for i in range(hist.size - 2):
               
        hist_scale[i]*=scale

    return hist_scale

def get_hist(obj,range=(132,4195),bins=10):
    return obj.to_hist()[range[0]:range[1]][hist.rebin(bins)]


def get_data(path,spectrum="mul_surv",det_type="all",r=(0,4000),b=10):
    """Get the histogram (PDF) and the number of simulated primaries (with uproot)"""

    file = uproot.open(path)
        
    if "{}/{}".format(spectrum,det_type) in file:
        hist = file["{}/{}".format(spectrum,det_type)]
        hist = get_hist(hist,range=r,bins=b)
        
    else:
        raise ValueError("Error: {}/{} not in {}".format(spectrum,det_type,path))
    return hist


def get_pdf_and_norm(path,spectrum="mul_surv",det_type="all",r=(0,4000),b=10):
    """Get the histogram (PDF) and the number of simulated primaries (with uproot)"""

    file = uproot.open(path)
        
    if "{}/{}".format(spectrum,det_type) in file:
        hist = file["{}/{}".format(spectrum,det_type)]
        hist = get_hist(hist,range=r,bins=b)
        N  = int(file["number_of_primaries"])
        
    else:
        raise ValueError("Error: {}/{} not in {}".format(spectrum,det_type,path))
    return hist,N


def get_counts_minuit(counts,energies):
    """A basic counting analysis implemented in Minuit"""
    cost = create_counting_likelihood(counts,np.diff(energies))
    bins=np.diff(energies)
    Ns_guess = abs(counts[1]-bins[1]*counts[0]/bins[0])+10

    Nb_guess=counts[0]
    guess=(Ns_guess,Nb_guess)
    m = Minuit(cost, *guess)
    m.limits[1]=(0,1.5*Nb_guess+10)
    m.limits[0]=(0,2*Ns_guess+10)
    m.migrad()
  
    if (m.valid):
        m.minos()
        elow=abs(m.merrors["Ns"].lower)
        ehigh=abs(m.merrors["Ns"].upper)
    else:
        elow = m.errors["Ns"]
        ehigh=m.errors["Ns"]
    #plot_counting_calc(energies,counts,m.values)

    return m.values["Ns"],elow,ehigh

def get_peak_counts(peak,name_peak,data_file,livetime=1,spec="mul_surv",size=5):
    """Get the counts in a peak in the data"""

    regions = {name_peak:(peak-size,peak+size)}
    left_sideband ={name_peak:(peak-3*size,peak-size)}
    right_sideband ={name_peak:(peak+size,peak+3*size)}

    det_types,name,Ns = get_det_types("all")

    #print(json.dumps(det_types,indent=1))
    energies= np.array([left_sideband[name_peak][0],left_sideband[name_peak][1],
                        regions[name_peak][1],right_sideband[name_peak][1]])

    data_counts = get_data_counts_total(spec,det_types,regions,data_file,key_list=[name_peak])
    data_counts_right = get_data_counts_total(spec,det_types,right_sideband,data_file,key_list=[name_peak])
    data_counts_left = get_data_counts_total(spec,det_types,left_sideband,data_file,key_list=[name_peak])

    ### the efficiency calculation
    counts,low,high=get_counts_minuit(np.array([data_counts_left["all"][name_peak],data_counts["all"][name_peak],data_counts_right["all"][name_peak]]),
                    
                    energies
                        )
    
    return counts/livetime,low/livetime,high/livetime

def get_eff_minuit(counts_total,counts_after,energies):
    """ Get the efficiency with Minuit the likelihood is just a binned one"""

    cost = create_efficiency_likelihood(counts_after,counts_total-counts_after)
    Ns_guess = counts_total[1]
    Nb_guess= counts_total[0]

    eff_s = counts_after[1]/counts_total[1]

    eff_b=counts_after[0]/counts_total[0]
    guess=(Ns_guess,Nb_guess,eff_s,eff_b)
    m = Minuit(cost, *guess)
    m.limits[1]=(0,1.5*Nb_guess+10)
    m.limits[0]=(0,2*Ns_guess+10)
    m.limits[2]=(0,1)
    m.limits[3]=(0,1)

    m.migrad()
    m.minos()
    elow=abs(m.merrors["eff_s"].lower)
    ehigh=abs(m.merrors["eff_s"].upper)
    plot_eff_calc(energies,counts_after,counts_total-counts_after,m.values)   
    return m.values["eff_s"],elow,ehigh

def get_data_counts(spectrum,det_type,regions,file):
    """Get the counts in the data in a range"""


    if "{}/{}".format(spectrum,det_type) in file:
        hist =file["{}/{}".format(spectrum,det_type)]
    else:
        data_counts={}
        warnings.warn("{}/{} not in data".format(spectrum,det_type))
        for region in regions.keys():
            data_counts[region]=0
        return data_counts
    
    data_counts={}
    hist=hist.to_hist()
    for region,rangel in regions.items():
        data=0
        for i in range(len(rangel)):
            data+= float(integrate_hist(hist,rangel[i][0],rangel[i][1]))
        
        data_counts[region]=data

    return data_counts


def name2number(meta,name:str):
    """Get the channel number given the name"""

    meta = LegendMetadata("../legend-metadata")

    chmap = meta.channelmap(datetime.now())

    if name in chmap:

        return f"ch{chmap[name].daq.rawid:07}"
    else:
        raise ValueError("Error detector {} does not have a number ".format(name))
    
def number2name(meta:LegendMetadata,number:str)->str:
    """Get the channel name given the channel number.

    Parameters:
        meta (LegendMetadata): An object containing metadata information.
        number (str): The channel number to be converted to a channel name.

    Returns:
        str: The channel name corresponding to the provided channel number.

    Raises:
        ValueError: If the provided channel number does not correspond to any detector name.

    """
 
    chmap = meta.channelmap(datetime.now())

    for detector, dic in meta.dataprod.config.on(datetime.now())["analysis"].items():
        if  f"ch{chmap[detector].daq.rawid:07}"==number:
            return detector
    raise ValueError("Error detector {} does not have a name".format(number))
            
def get_channel_floor(name:str,groups_path:str="cfg/level_groups_Sofia.json")->str:
    """Get the z group for the detector
    Parameters:
        name: (str) 
            - the channel name
        groups_path: (str, optional)
            - the path to a JSON file containing the groupings, default level_groups.json
    Returns:
        - str: The group a given channel corresponds to
    Raises:
        - Value error if the channel isnt in the json file
    """

    with open(groups_path, 'r') as json_file:
        level_groups =json.load(json_file)

    for key,chan_list in level_groups.items():
        if name in chan_list:
            return key
        
    raise ValueError("Channel {} has no position ".format(name))



def get_channels_map():
    """ Get the channels map
    Parameters:
        None
    Returns:
        - a dictonary of string channels of the form

        {
        1:
        {      
            ["channel1","channel2", etc (the names V002 etc)
        },
        }

        - a dictonary of types like

        {   
        1:
        {
            ["icpc","bege",...]
        }
        
    """

    meta = LegendMetadata("../legend-metadata/")
 
    ### 1st July should be part of TAUP dataset
    time = datetime(2023, 7, 1, 00, 00, 00, tzinfo=timezone.utc)
    chmap = meta.channelmap(datetime.now())



    string_channels={}
    string_types={}
    for string in range(1,12):
        if (string==6):
            continue
        channels_names = [
            f"ch{chmap[detector].daq.rawid:07}"
            for detector, dic in meta.dataprod.config.on(time)["analysis"].items()
            if dic["processable"] == True
            and chmap[detector]["system"] == 'geds'
            and chmap[detector]["location"]["string"] == string
        ]
        channels_types = [
            chmap[detector]['type']
            for detector, dic in meta.dataprod.config.on(time, system="cal")["analysis"].items()
            if dic["processable"] == True
            and chmap[detector]["system"] == 'geds'
            and chmap[detector]["location"]["string"] == string
        ]
       
        string_channels[string]=channels_names
        string_types[string]=channels_types

    return string_channels,string_types

def get_exposure(group:list,periods:list=["p03","p04","p06","p07"])->float:
    """
    Get the livetime for a given group a list of strs of channel names or detector types or "all"
    Parameters:
        group: list of str
        periods: list of periods to consider default is 3,4,6,7 (vancouver dataset)
    Returns:
        - a float of the exposure (summed over the group)
    """

    meta = LegendMetadata("../legend-metadata")

    with open("cfg/analysis_runs.json",'r') as file:
        analysis_runs=json.load(file)
    ### also need the channel map
  
    groups_exposure=np.zeros(len(group))

    for period in periods:

        if period not in analysis_runs.keys():
            continue
        ## loop over runs
        for run,run_dict in meta.dataprod.runinfo[period].items():
            
            ## skip 'bad' runs
            if ( run in analysis_runs[period]):

                ch = meta.channelmap(run_dict["phy"]["start_key"])

                for i in range(len(group)):
                    item =group[i]

                    if item=="all":
                        geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and ch[_name]["analysis"]["usability"] in ["on","no_psd"]]
                    elif ((item=="bege")or(item=="icpc")or (item=="coax")or (item=="ppc")):
                        geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and ch[_name]["type"]==item and ch[_name]["analysis"]["usability"] in ["on","no_psd"]]
                    else:
                        geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and f"ch{ch[_name]['daq']['rawid']}"==item and ch[_name]["analysis"]["usability"] in ["on","no_psd"]]


                    ## loop over detectors
                    for det in geds_list:
                        mass = ch[det].production.mass_in_g/1000
                        if "phy" in run_dict:
                            groups_exposure[i] += mass * (run_dict["phy"]["livetime_in_s"]/(60*60*24*365.25
                                                                                                ))
   
    return np.sum(groups_exposure)



def get_livetime(verbose:bool=True)->float:
    """ Get the livetime from the metadata
    Parameters:
        verbose: (bool), default True a bool to say if the livetime is printed to the screen
    Returns:
        - float : the livetime in s
    
    """

    meta = LegendMetadata("../legend-metadata")

    with open("cfg/analysis_runs.json",'r') as file:
        analysis_runs=json.load(file) 

    taup=0
    vancouver=0

    ### loop over periods
    for period in meta.dataprod.runinfo.keys():
        livetime_tot =0

        ## loop over runs
        for run in meta.dataprod.runinfo[period].keys():
            
            ## skip 'bad' runs
            if (period in analysis_runs.keys() and run in analysis_runs[period]):

                if "phy" in meta.dataprod.runinfo[period][run].keys():
                    time = meta.dataprod.runinfo[period][run]["phy"]["livetime_in_s"]
                    livetime_tot+=time

        ### sum the livetimes
        if (period=="p03" or period=="p04"):
            taup +=livetime_tot
            vancouver+=livetime_tot
        if (period=="p06" or period=="p07"):
            vancouver+=livetime_tot

    if (verbose):
        print("Taup livetime = {}".format(taup))
        print("Vancouver livetime = {}".format(vancouver))

    return vancouver


def get_det_types(group_type:str,string_sel:int=None,det_type_sel:str=None,level_groups:str="cfg/level_groups_Sofia.json")->(dict,str,list):
    """
    Extract a dictonary of detectors in each group

    Parameters:
        grpup_type (str): 
            The type of grouping (sum,types,string,chan,floor or all)
        string_sel (int, optional): 
            An integer representing the selection of a particular string. Default is None ie select all strings
        det_type_sel (str, optional): 
            A string representing the selection of a particular detector type. Default is None (select all det types)
        level_groups (str, optional) a str of where to find a json file to get the groupings for floor default level_groups.json
    Returns:
        tuple: A tuple containing three elements:
            - A dictionary containing the channels in each group, with the structure
                "group":
                {
                    "names": ["det1","det2", ...],
                    "types": ["icpc","bege",....],
                    "exposure": XXX, 
                },
            - A string representing the name of this grouping
            - A list of the number of channels per grouping - only used for the chans option

    Example:
        det_info, selected_det_type, selected_det = get_det_types("sum")
    """


    ### extract the meta data
    meta = LegendMetadata("../legend-metadata")
    Ns=[]

    ### just get the total summed over all channels
    if (group_type=="sum" or group_type=="all"):
        det_types={"all":{"names":["icpc","bege","ppc","coax"],"types":["icpc","bege","ppc","coax"]}}
        namet="all"
    elif (group_type=="types"):
        det_types={"icpc": {"names":["icpc"],"types":["icpc"]},
               "bege": {"names":["bege"],"types":["bege"]},
               "ppc": {"names":["ppc"],"types":["ppc"]},
               "coax": {"names":["coax"],"types":["coax"]}
            }
        namet="by_type"
    elif (group_type=="string"):
        string_channels,string_types = get_channels_map()
        det_types={}

        ### loop over strings
        for string in string_channels.keys():
            det_types[string]={"names":[],"types":[]}

            for dn,dt in zip(string_channels[string],string_types[string]):
                
                if (det_type_sel==None or det_type_sel==dt):
                    det_types[string]["names"].append(dn)
                    det_types[string]["types"].append(dt)

        namet="string"

    elif (group_type=="chan"):
        string_channels,string_types = get_channels_map()
        det_types={}
        namet="channels"
        Ns=[]
        N=0
        for string in string_channels.keys():

            if (string_sel==None or string_sel==string):
                chans = string_channels[string]
                types = string_types[string]

                for chan,type in zip(chans,types):
                    if (det_type_sel==None or type==det_type_sel):
                        det_types[chan]={"names":[chan],"types":[type]}
                        N+=1
            Ns.append(N)

    elif (group_type=="floor"):
   
        string_channels,string_types = get_channels_map()
        groups=["top","mid_top","mid","mid_bottom","bottom"]
        det_types={}
        namet="floor"
        for group in groups:
            det_types[group]={"names":[],"types":[]}
        
        for string in string_channels.keys():
            channels = string_channels[string]
            
            ## loop over channels per string
            for i in range(len(channels)):
                chan = channels[i]
                name = number2name(meta,chan)
                group = get_channel_floor(name,groups_path=level_groups)
                if (string_sel==None or string_sel==string):
                    
                    if (det_type_sel==None or string_types[string][i]==det_type_sel):
                        det_types[group]["names"].append(chan)
                        det_types[group]["types"].append(string_types[string][i])
    else:
        raise ValueError("group type must be either floor, chan, string sum all or types")
    
    ### get the exposure
    ### --------------
    for group in det_types:
        list_of_names= det_types[group]["names"]
        exposure = get_exposure(list_of_names)

        det_types[group]["exposure"]=exposure

    return det_types,namet,Ns

def select_region(histo,cuts=[]):
    """Select a region of the histo"""
    histo_out=copy.deepcopy(histo)
    energy_2 = histo_out.axes.centers[0]
    energy_1 = histo_out.axes.centers[1]
    matrix_e2, matrix_e1 = np.meshgrid(energy_2, energy_1, indexing='ij')
    energy_1=energy_1[0]
    energy_2=energy_2.T[0]

    matrix_sum = energy_2[:, np.newaxis] + energy_1
    matrix_diff = -energy_2[:,np.newaxis]+energy_2
    logic =  np.ones_like(matrix_sum,dtype="bool")

    for cut in cuts:
    
        if cut["var"]=="e1":
            matrix= matrix_e1
        elif cut["var"]=="e2":
            matrix=matrix_e2
        elif cut["var"]=="diff":
            matrix= matrix_diff
        else:
            matrix=matrix_sum
        
        if (cut["greater"]==True):
            logic_tmp = matrix>cut["val"]
        else:
            logic_tmp = matrix<=cut["val"]
      
        logic=(logic) & (logic_tmp)
    
    w,x,y=histo_out.to_numpy()
    w_new = w
    w_new[~logic]=0
  
    for i in range(len(histo_out.axes.centers[0])):
        for j in range(len(histo_out.axes.centers[1])):
            histo_out[i,j]=w_new[i,j]
    return histo_out


def project_sum(histo):
    """ Take a 2D histogram and create a 1D histogram with the sum of the bin contents
        Note: This is in principle impossible here an approximation is used where a sum of a -- a+1 and b-- b+1 is split evenly between bin a+b -- a+b+1 and a+b+1 --- a+b +2
        If you want a proper summed histo you should build it directly from the raw data
    """
    w,x,y=histo.to_numpy()
    
    hist_energy_sum =( Hist.new.Reg(int((len(x)-2)*2)+2, 0, x[-1]+y[-1]).Double())
  
    rows, cols = w.shape

    # Create an array of indices for each element
    indices = np.arange(rows) + np.arange(cols)[:, None]
  
    # Use numpy's bincount to calculate diagonal sums
    diagonal_sums = np.bincount(indices.flatten(), weights=w.flatten())
 
    for i in range(hist_energy_sum.size-3):
        hist_energy_sum[i]+=diagonal_sums[i]
  
    return hist_energy_sum



def fast_variable_rebinning(x, y, weights, edges_x, edges_y):
    # Determine the bin indices for each (x, y) pair

    indices_x = np.searchsorted(edges_x, x[0:-1]+0.1) - 1
    indices_y = np.searchsorted(edges_y, y[0:-1]+0.1) - 1
    
    flat_indices = indices_x[:, np.newaxis] * len(edges_y) + indices_y
    #flat_indices=((indices_x+1)[:,np.newaxis]*(indices_y+1)-1)
    fig, axes = lps.subplots(1, 1, figsize=(5,3), sharex=True, gridspec_kw = {'hspace': 0})
    
    flat_indices=flat_indices.flatten()
   

    # Calculate the total number of bins
    num_bins_x = len(edges_x) 
    num_bins_y = len(edges_y) 
    
    # Create a 2D histogram using bin indices and weights
    hist = np.zeros((num_bins_x, num_bins_y)).flatten()

    np.add.at(hist, flat_indices, weights.flatten())

  
    wrb = hist.reshape((num_bins_x,num_bins_y))
  
    histo_var =( Hist.new.Variable(edges_x).Variable(edges_y).Double())

    for i in range(int(len(np.hstack(histo_var.axes.centers[0])))):
        for j in range(int(len(histo_var.axes.centers[1][0]))):
            histo_var[i,j]=wrb[i,j]

    return histo_var

def variable_rebin(histo,edges):
    """ Perform a variable rebinning of a hist object"""
    histo_var =( Hist.new.Variable(edges).Double())
    for i in range(histo.size-2):
        cent = histo.axes.centers[0][i]
    
        histo_var[cent*1j]+=histo.values()[i]

    return histo_var

def variable_rebin_2D(histo,edges_x,edges_y):
    histo_var =( Hist.new.Variable(edges_x).Variable(edges_y).Double())
    print(histo.axes.centers)
    print(histo.values())
    print(np.hstack(histo.axes.centers[0]))
    xarr =np.hstack(histo.axes.centers[0])
    vals=histo.values()
    for i in range(int(len(np.hstack(histo.axes.centers[0])))):
        if (i%100==0):
            print("Progreess = {:02f} %".format(100*i/len(np.hstack(histo.axes.centers[0]))))
        for j in range(int(len(histo.axes.centers[1][0]))):

            cent_x = xarr[i]
            cent_y=histo.axes.centers[1][0][j]
            histo_var[cent_x*1j,cent_y*1j]+=vals[i,j]

    return histo_var

def normalise_histo_2D(histo,factor=1):
    widths_x= np.diff(np.hstack(histo.axes.edges[0]))
    widths_y= np.diff(histo.axes.edges[1][0])
    print(histo.axes.edges)
    print(widths_x,widths_y)
    for i in range(int(len(np.hstack(histo.axes.centers[0])))):
        if (i%100==0):
            print("Progreess = {:02f} %".format(100*i/len(np.hstack(histo.axes.centers[0]))))
        for j in range(int(len(histo.axes.centers[1][0]))):
            if i<4 or j<4:
                histo[i,j]=0
            else:
                histo[i,j]/=(widths_x[i]*widths_y[j])
                histo[i,j]*=factor
                
    return histo

def normalise_histo(hist,factor=1):
    """ Normalise a histogram into units of counts/keV"""

    widths= np.diff(hist.axes.edges[0])

    for i in range(hist.size-2):
        hist[i]/=widths[i]
        hist[i]*=factor
    return hist