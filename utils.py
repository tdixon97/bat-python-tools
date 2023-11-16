import re
import pandas as pd
import uproot
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
import matplotlib.pyplot as plt
import numpy as np

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
    list_new=[]
    for text in list_str:
        modified_string =text.replace("Pb214", "$^{214}$Pb")
        modified_string=modified_string.replace("Bi214","$^{214}$Bi")
        modified_string=modified_string.replace("Tl208","$^{214}$Bi")
        modified_string=modified_string.replace("K40","$^{40}$K")
        modified_string=modified_string.replace("K42","$^{42}$K")
        modified_string=modified_string.replace("Bi212","$^{212}$Bi")

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





def plot_two_dim(varx:np.ndarray,vary:np.ndarray,rangex:tuple,rangey:tuple,titlex:str,titley:str,title:str,bins:tuple):
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

    fig, axes = lps.subplots(1, 1, figsize=(4,6), sharex=True, gridspec_kw = {'hspace': 0})

    h = axes.hist2d(varx,vary, bins=bins, cmap='viridis', range=[rangex,rangey], cmin=1)

    fig.colorbar(h[3], ax=axes, label='Counts')
    axes.set_xlabel(titlex)
    axes.set_ylabel(titley)
    axes.set_title(title)

  

    correlation_coefficient = np.corrcoef(varx, vary)[0, 1]

    # Annotate the plot with correlation coefficient
    axes.annotate("Correlation = {:0.2f}".format(correlation_coefficient), (0.6, 0.88), xycoords="axes fraction", fontsize=10)
    plt.show()
    return correlation_coefficient

def plot_correlation_matrix(corr_matrix:np.ndarray,title:str):
    """
    Plots a correlation matrix 

    Parameters:
    - corr_matrix (numpy.ndarray): 2D NumPy array representing the correlation matrix.
    """

    # Set up the matplotlib figure
    fig, axes = lps.subplots(1, 1, figsize=(4, 4), sharex=True, gridspec_kw = {'hspace': 0})


    # Create a heatmap
    cax = axes.matshow(corr_matrix, cmap='coolwarm',vmin=-1,vmax=1)
    axes.set_title(title)
    
    # Add a colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Correlation')
    # Show the plot
    plt.show()
