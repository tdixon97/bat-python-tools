import re
import pandas as pd
import uproot
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc


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
    

def ttree2df(filename:str,tree_name:str)->pd.DataFrame:
    """Use uproot to import a TTree and save to a dataframe
    Parameters:
        - filename: file to open
        - tree_name: Which TTree to look at
    Returns:
        Pandas DataFrame of the data
    """

    file = uproot.open(filename)

    # Access the TTree inside the file
    tree = file[tree_name]

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
                        y2=None,ylow2=None,yhigh2=None,label1=None,label2=None,extra=None,do_comp=0):
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
    fig, axes = lps.subplots(1, 1, figsize=(6, height), sharex=True, gridspec_kw = {'hspace': 0})

    xin=np.arange(len(labels))
    
    if (do_comp==False):
        axes.errorbar(y=xin+0.15*len(y)/30,x=y,xerr=[ylow,yhigh],fmt="o",color=vset.blue,ecolor=vset.cyan,markersize=1)
    else:
        axes.errorbar(y=xin+0.15*len(y)/30,x=y,xerr=[ylow,yhigh],fmt="o",color=vset.blue,ecolor=vset.cyan,markersize=1,
                      label=label1)
        axes.errorbar(y=xin-0.15*len(y2)/30,x=y2,xerr=[ylow2,yhigh2],fmt="o",color=vset.red,ecolor=vset.orange,markersize=1,
                      label=label2)


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
        axes.set_xlim(0.1,upper)
    elif (obj=="scaling_factor"):
        axes.set_xlabel("Scaling factor [1/yr] ")
        axes.set_xlim(1E-7,upper)
    elif obj=="parameter":
        axes.set_xlabel("Decays / yr [1/yr]")
        axes.set_xlim(0.1,upper)
 
    axes.set_yticks(axes.get_yticks())
    
    axes.set_yticklabels([val for val in labels], fontsize=8)
    plt.xscale("log")
    plt.grid()
    leg=fig.legend(loc='upper right',bbox_to_anchor=(1.0, .9),frameon=True, facecolor='white')
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
    
    for key in names:
       
        if (key.find("Bi212")!=-1):
            index_Th.append(i)
        elif(key.find("Bi214")!=-1):
            index_U.append(i)
        elif(key.find("K")!=-1):
        
            index_K.append(i)    
        i=i+1

    return {"U":index_U,"Th":index_Th,"K":index_K}


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