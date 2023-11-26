import re
import pandas as pd
import uproot
import copy
from legend_plot_style import LEGENDPlotStyle as lps
from datetime import datetime, timezone

lps.use('legend')
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
import json
from legendmeta import LegendMetadata
import warnings

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


def get_efficiencies(cfg,spectrum,det_type,regions,pdf_path,name,spectrum_fit=""):
    """ Get the efficiencies"""

    effs={"full":{},"2nu":{},"K40":{},"K42":{},"Tl compton":{},"Tl peak":{}}

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
    print(json.dumps(meta.dataprod.runinfo,indent=1))

    print(meta.dataprod.runinfo.keys())
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