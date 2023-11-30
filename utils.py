import re
import pandas as pd
import uproot
import copy
import hist

from legend_plot_style import LEGENDPlotStyle as lps
from datetime import datetime, timezone
from scipy.stats import poisson
lps.use('legend')
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
import json
from legendmeta import LegendMetadata
import warnings
from iminuit import Minuit, cost
from scipy.stats import expon
from scipy.stats import truncnorm
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit

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
        

def format_latex(list_str):
    """Format a string as latex with the radioisotopes formatted"""
    list_new=[]
    for text in list_str:
        modified_string =text.replace("Pb214", "$^{214}$Pb")
        modified_string =modified_string.replace("2vbb", "$2\\nu\\beta\\beta$")

        modified_string=modified_string.replace("Bi214","Bi")
        modified_string=modified_string.replace("Tl208","$^{208}$Tl")
        modified_string=modified_string.replace("K40","$^{40}$K")
        modified_string=modified_string.replace("K42","$^{42}$K")
        modified_string=modified_string.replace("Bi212","$^{212}$Bi")
        modified_string=modified_string.replace("Ac228","$^{228}$Ac")
        modified_string=modified_string.replace("Ar39","$^{39}$Ar")

        list_new.append(modified_string)
    return list_new
    

def ttree2df(filename:str,data:str)->pd.DataFrame:
    """Use uproot to import a TTree and save to a dataframe
    Parameters:
        - filename: file to open
        - tree_name: Which TTree to look at
    Returns:
        Pandas DataFrame of the data
    """

    file = uproot.open(filename)

    # Access the TTree inside the file

    for key in file.keys():
        if (data in key):
            tree = file[key]

    # Get the list of branches in the TTree
    branches = tree.keys()

    # Define a dictionary to store the branch data
    data = {}

    # Loop through each branch and extract the data

    for branch_name in branches:
        # Use uproot to read the branch into a NumPy array
        data[branch_name] = tree[branch_name].array()

    # Create a Pandas DataFrame from the dictionary
    return pd.DataFrame(data)





def plot_two_dim(varx:np.ndarray,vary:np.ndarray,rangex:tuple,rangey:tuple,titlex:str,titley:str,title:str,bins:tuple,show=False,save=""):
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
        plt.savefig(save)
        
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
    sort_m=-matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
                if (i>=j):
                    sort_m[i,j]=0
    
    indices_nth_largest = np.argsort(sort_m.flatten())[-(n+1)]
    
    row_indices, col_indices = np.unravel_index(indices_nth_largest, matrix.shape)  
    
    return sort_m[row_indices,col_indices],row_indices,col_indices

def plot_corr(df,i,j,labels,save):
    key1=df.keys()[i]
    key2=df.keys()[j]
    x=np.array(df[key1])
    y=np.array(df[key2])
    rangex=(0,max(x))
    rangey=(0,max(y))
    bins=(100,100)
        
    cor=plot_two_dim(x,y,rangex,rangey,"{} [1/yr]".format(labels[i]),
                                                    "{} [1/yr]".format(labels[j]),
                                                    "{} vs {}".format(labels[i],labels[j]),bins,True,save)
    


def make_error_bar_plot(indexs,labels:list,y:np.ndarray,ylow:np.ndarray,yhigh:np.ndarray,data,name_out,obj,
                        y2=None,ylow2=None,yhigh2=None,label1=None,label2=None,extra=None,do_comp=0,low=0.1,scale="log",upper=0):
    """
    Make the error bar plot
    """
    indexs=np.array(indexs,dtype="int")
    vset = tc.tol_cset('vibrant')
    labels=np.array(labels)
    y=y[indexs]
    labels=labels[indexs]
    ylow=ylow[indexs]
    yhigh=yhigh[indexs]

    if (do_comp==True):
        
        ylow2=ylow2[indexs]
        yhigh2=yhigh2[indexs]
        y2=y2[indexs]

    height= 2+4*len(y)/29
    fig, axes = lps.subplots(1, 1, figsize=(8, height), sharex=True, gridspec_kw = {'hspace': 0})

    xin=np.arange(len(labels))
    
    if (do_comp==False):
        axes.errorbar(y=xin+0.15*len(y)/30,x=y,xerr=[ylow,yhigh],fmt="o",color=vset.blue,ecolor=vset.cyan,markersize=1)
    else:
        axes.errorbar(y=xin+0.15*len(y)/30,x=y,xerr=[ylow,yhigh],fmt="o",color=vset.blue,ecolor=vset.cyan,markersize=1,
                      label=label1)
        axes.errorbar(y=xin-0.15*len(y2)/30,x=y2,xerr=[ylow2,yhigh2],fmt="o",color=vset.red,ecolor=vset.orange,markersize=1,
                      label=label2)

    if (upper==0):
        upper = np.max(y+1.5*yhigh)
        if (do_comp==True):
            upper=max(upper,np.max(y2+1.5*yhigh2))
            upper=upper*5
    axes.set_yticks(xin, labels)

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
    
    axes.set_yticklabels([val for val in labels], fontsize=8)
    axes.set_xscale(scale)
    plt.grid()
    leg=axes.legend(loc='best',edgecolor="black",frameon=True, facecolor='white',framealpha=1)
    leg.set_zorder(10)    

    #plt.show()
    if (do_comp==True):
        name_out="comp_{}_to_{}_{}".format(label1,label2,extra)
    if (obj!="scaling_factor"):     
        plt.savefig("plots/fit_results/fit_results_{}_{}_{}.pdf".format(data,obj,name_out))
    else:
        plt.savefig("plots/fit_results/fit_results_{}_{}.pdf".format(obj,name_out))


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
        elif(key.find("K")!=-1):
        
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

def get_total_efficiency(det_types,cfg,spectrum,regions,pdf_path,det_sel="all"):
    eff_total={}
    ### creat the efficiency maps (per dataset)
    for det_name, det_info in det_types.items():
        
        det_list=det_info["names"]
        effs={}
        for key in regions:
            effs[key]={}

        for det,named in zip(det_list,det_info["types"]):
            eff_new,good = get_efficiencies(cfg,spectrum,det,regions,pdf_path,named,"mul_surv")
            if (good==1 and (named==det_sel or det_sel=="all")):
                effs=sum_effs(effs,eff_new)

        eff_total[det_name]=effs

    return eff_total
def get_efficiencies(cfg,spectrum,det_type,regions,pdf_path,name,spectrum_fit=""):
    """ Get the efficiencies"""

    effs={}
    for key in regions:
        effs[key]={}
  
    if (spectrum_fit==""):
        spectrum_fit=spectrum

    for key,region in regions.items():
        effs[key]["2vbb_bege"]=0
        effs[key]["2vbb_coax"]=0
        effs[key]["2vbb_ppc"]=0
        effs[key]["2vbb_icpc"]=0

    for key in cfg["fit"]["theoretical-expectations"].keys():
        if ".root" in key:
            filename = key

    if "{}/{}".format(spectrum_fit,"icpc") in cfg["fit"]["theoretical-expectations"][filename]:
    
        icpc_comp_list=cfg["fit"]["theoretical-expectations"][filename]["{}/{}".format(spectrum_fit,name)]["components"]
    else:
        warnings.warn("{}/{} not in MC PDFs".format(spectrum_fit,det_type))
        return effs,0
    comp_list = copy.deepcopy(icpc_comp_list)
    
    for comp in comp_list:
        for key in comp["components"].keys():
            par = key
        ## now open the file
            
        file = uproot.open(pdf_path+comp["root-file"])
        
        if "{}/{}".format(spectrum,det_type) in file:
            hist = file["{}/{}".format(spectrum,det_type)]
            N  = int(file["number_of_primaries"])
            hist = hist.to_hist()

            for key,region in regions.items():

                eff= float(integrate_hist(hist,region[0],region[1]))
                
                effs[key][par]=eff/N
        else:

            warnings.warn("{}/{} not in MC PDFs".format(spectrum,det_type))
            for key,region in regions.items():
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


def create_counting_likelihood(data):
    """Create the likelihood function"""

    def likelihood(Ns,Nb):
        """The likelihood function"""

        logL=0
        preds_surv =np.array([Nb,Nb+Ns,Nb])
      

        logL+=sum(poisson.logpmf(data, preds_surv))
       
        return -logL
    return likelihood



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
    elif ("exp" in prior):
        split_parts = prior.split("/")

        # Extract the upper limit from the exponenital
        if len(split_parts) > 1:
            upper_limit = float( split_parts[1][:-1])
            rv= expon(scale=upper_limit/2.3)
            high = 2*upper_limit
        else:
            raise ValueError("exp prior doesnt contain the part /UpperLimit) needed")
    else:
        raise ValueError("Only supported priors are gaus and exp currently")
    
    return rv,high



def plot_mc_no_save(axes,pdf,name="",linewidth=0.6):
    
    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("counts/10 keV")

    axes.set_yscale("log")
    axes.set_xlim(0,4000)
    pdf.plot(ax=axes, linewidth=linewidth, yerr=False,flow= None,histtype="step",label=name)
    #pdf.plot(ax=axes, linewidth=0.8, yerr=False,flow= None,histtype="fill",alpha=0.15)

    return axes

def compare_mc(max_energy,pdfs,decay,order,norm_peak,pdf,xlow,xhigh,scale,linewidth=0.6):
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
    
def plot_mc(pdf,name,save_pdf,data=None,range_x=(500,4000),range_y=(0.01,5000),pdf2=None):
    vset = tc.tol_cset('vibrant')

    fig, axes = lps.subplots(1, 1,figsize=(5, 4), sharex=True, gridspec_kw = { "hspace":0})
    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("counts/10 keV")
    axes.set_title(name)
    axes.set_yscale("log")
    axes.set_xlim(range_x)
    axes.set_ylim(range_y)
    if (data is not None):
        data.plot(ax=axes,yerr=False,flow=None,histtype="fill",alpha=0.3,label="Data",color=vset.blue)
    pdf.plot(ax=axes, linewidth=1,color=vset.red, yerr=False,flow= None,histtype="step",label="Best fit from screening")
    if (pdf2 is not None):
        pdf2.plot(ax=axes, linewidth=1,color=vset.red,alpha=0.6, yerr=False,flow= None,histtype="step",label="90 pct upper limit")

    axes.legend(loc='best',edgecolor="black",frameon=True, facecolor='white',framealpha=1)    

    save_pdf.savefig()

    plt.close()

def vals2hist(vals,hist):
    for i in range(hist.size - 2):
        hist[i]=vals[i]
    return hist

def slow_convolve(priors,pdfs,rvs,n_samp=10000):
    """
    
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





def fast_convolve():


    ### create the data array

    S= np.ones((1000,10,10000))
    #S[:,0,0]=np.linspace(0,1000,1000)
    print(S)
    
    

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
    cost = create_counting_likelihood(counts)
    Ns_guess = counts[1]

    Nb_guess=counts[0]
    guess=(Ns_guess,Nb_guess)
    m = Minuit(cost, *guess)
    m.limits[1]=(0,1.5*Nb_guess+10)
    m.limits[0]=(0,2*Ns_guess+10)

    m.migrad()
    m.minos()
    elow=abs(m.merrors["Ns"].lower)
    ehigh=abs(m.merrors["Ns"].upper)

    return m.values["Ns"],elow,ehigh

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
    for region,range in regions.items():
        data= float(integrate_hist(hist,range[0],range[1]))
        
        data_counts[region]=data

    return data_counts


def name2number(meta,name:str):
    """Get the channel number given the name"""

    meta = LegendMetadata()

    chmap = meta.channelmap(datetime.now())

    if name in chmap:

        return f"ch{chmap[name].daq.rawid:07}"
    else:
        raise ValueError("Error detector {} does not have a number ".format(name))
def number2name(meta,number:str):
    """ Get the name given the number"""
    
 
    chmap = meta.channelmap(datetime.now())


    for detector, dic in meta.dataprod.config.on(datetime.now())["analysis"].items():
        if  f"ch{chmap[detector].daq.rawid:07}"==number:
            return detector
    raise ValueError("Error detector {} does not have a name".format(number))
            
def get_channel_floor(name:str):
    """Get the z group for the detector"""

    level_groups = {"top":["V02160A", "B00035C",  "B00032B", "B00000B", "V08682B", "V02162B", 'B00089C', 'B00002A', 'B00000D', 'B00000A' ,
                         "C000RG1",  'B00091C', "B00000C", 'B00089D', 'B00089A', 'B00002C', 'B00002B'] , 
                "mid_top":['V02160B', 'V08682A', 'V02166B', 
                           'V05261B', 'C000RG2', 'P00574B', 'B00061A', 'C00ANG4', 'B00091A', 'B00032C', 'B00032A',
                          'P00665A', 'B00061C','B00091D','B00032D','P00538B', 'B00076C','B00035A'], 
                "mid":['V09372A', 'V04199A', 'V05266A', 'C00ANG3', 'V09374A', 'V04545A','V00048B', 'V05266B', 'C00ANG5', 'P00698A','V09724A', 'V05267B','V00050A','P00537A','P00573B',
                      'P00712A','B00079B','V00050B','P00538A','B00035B','P00575A','P00909C','B00079C','P00573A','B00061B','P00574C','V01386A','P00661C','B00089B','P00661A','V00048A'], 
                "mid_bottom":['V05268B','C00ANG2','V07646A','V05612A','V00074A','V01387A','P00661B','V05261A','P00574A','V01403A','P00662C','P00662A','V01404A','P00664A','V01240A','P00662B',
                             'V01406A','P00665C','V01389A', 'P00665B', ], 
                "bottom":['V07302B','V07647A','V04549A','V07647B','V07298B','V05268A','V07302A','V01415A', 'P00748B', 'V05267A', 'P00748A', 'P00909B', 'V05612B', 'P00698B', 'B00091B']}
    
    for key,chan_list in level_groups.items():
        if name in chan_list:
            return key
    raise ValueError("Channel {} has no position ".format(name))

def get_channels_map():
    """ Get the channels map"""

    meta = LegendMetadata()
 
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


def get_livetime():
    """ Get the livetime from the metadata"""

    meta = LegendMetadata()

    bad_runs=[("p04","r006"),("p07","r000")]
 

    taup=0
    vancouver=0
    for period in meta.dataprod.runinfo.keys():
        livetime_tot =0
        for run in meta.dataprod.runinfo[period].keys():
            
            if ((period,run) in bad_runs):
                continue
            if "phy" in meta.dataprod.runinfo[period][run].keys():
                time = meta.dataprod.runinfo[period][run]["phy"]["livetime_in_s"]
                livetime_tot+=time
                print("For run {} {} time = {}".format(run,period,time))

        print("For period  {} livetime = {}".format(period,livetime_tot))
        if (period=="p03" or period=="p04"):
            taup +=livetime_tot
            vancouver+=livetime_tot
        if (period=="p06" or period=="p07"):
            vancouver+=livetime_tot

    print("Taup livetime = {}".format(taup))
    print("Vancouver livetime = {}".format(vancouver))



def get_det_types(det_type,string_sel=0,det_sel=None):
    """Get a dictonary of detector types"""

    meta = LegendMetadata()

    if (det_type!="sum" and det_type!="str" and det_type!="chan" and det_type!="floor"):
        det_types={"icpc": {"names":["icpc"],"types":["icpc"]},
               "bege": {"names":["bege"],"types":["bege"]},
               "ppc": {"names":["ppc"],"types":["ppc"]},
               "coax": {"names":["coax"],"types":["coax"]}
            }
        namet="by_type"
    elif (det_type=="sum"):
        det_types={"all":{"names":["icpc","bege","ppc","coax"],"types":["icpc","bege","ppc","coax"]}}
        namet="all"
    
    elif(det_type=="str"):
        string_channels,string_types = get_channels_map()
        det_types={}
        for string in string_channels.keys():
            det_types[string]={"names":[],"types":[]}

            for dn,dt in zip(string_channels[string],string_types[string]):
                
                if (det_sel==None or det_sel==dt):
                    det_types[string]["names"].append(dn)
                    det_types[string]["types"].append(dt)

        namet="string"

    ## select by channel
    elif (det_type=="chan"):
        string_channels,string_types = get_channels_map()
        det_types={}
        namet="channels"
        for string in string_channels.keys():

            if (string_sel==0 or string_sel==string):
                chans = string_channels[string]
                types=string_types[string]

                for chan,type in zip(chans,types):
                    if (det_sel==None or type==det_sel):
                        det_types[chan]={"names":[chan],"types":[type]}

    ### select by floor
    elif (det_type=="floor"):
        string_channels,string_types = get_channels_map()
        groups=["top","mid_top","mid","mid_bottom","bottom"]
        det_types={}
        namet="floor"
        for group in groups:
            det_types[group]={"names":[],"types":[]}
        
        for string in string_channels.keys():
            channels = string_channels[string]
            for i in range(len(channels)):
                chan = channels[i]
                name = number2name(meta,chan)
                group = get_channel_floor(name)
                if (string_sel==0 or string_sel==string):
                    
                    if (det_sel==None or string_types[string][i]==det_sel):
                        det_types[group]["names"].append(chan)
                        det_types[group]["types"].append(string_types[string][i])

    return det_types,namet