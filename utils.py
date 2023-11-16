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





def plot_two_dim(varx:np.ndarray,vary:np.ndarray,rangex:tuple,rangey:tuple,titlex:str,titley:str,title:str,bins:tuple,show=False):
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

        h = axes.hist2d(varx,vary, bins=bins, cmap='viridis', range=[rangex,rangey], cmin=1)

        #fig.colorbar(h[3], ax=axes, label='Counts')
        axes.set_xlabel(titlex)
        axes.set_ylabel(titley)
        axes.set_title(title)

  

    correlation_coefficient = np.corrcoef(varx, vary)[0, 1]

    # Annotate the plot with correlation coefficient
    if (show==True):
        axes.annotate("Correlation = {:0.2f}".format(correlation_coefficient), (0.6, 0.88), xycoords="axes fraction", fontsize=10)
    
        plt.show()
        plt.close()
    return correlation_coefficient

def plot_correlation_matrix(corr_matrix:np.ndarray,title:str,save:str):
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
    #cbar.set_label('Correlation')
    plt.grid()
    # Show the plot
    plt.savefig(save)
    plt.show()
    


def plot_table(df,save):
    """ Plot a df as a table"""
    # Create a DataFrame


    # Plot a table
    fig, ax = plt.subplots(figsize=(2, 6))  # Adjust the figsize as needed
    ax.axis('off')  # Turn off axis labels

    # Create the table
    table_data = df.T.reset_index().values
    print(table_data)
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
    plt.show()