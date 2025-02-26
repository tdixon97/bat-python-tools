from __future__ import annotations

import copy
import math
import os
import re
from datetime import datetime, timezone
from warnings import simplefilter

import hist
import pandas as pd
import uproot
from legend_plot_style import LEGENDPlotStyle as lps
from scipy.stats import norm, poisson

lps.use("legend")
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
from hist import Hist
from iminuit import Minuit
from legendmeta import LegendMetadata
from scipy.stats import expon, truncnorm


def get_list_of_directories(file):
    """Get the list of directories inside a root file
    Parameters:
        - the file object from uproot
    Returns:
        - a list of str of directories
    """

    directories = [
        key for key in file.keys() if isinstance(file[key], uproot.reading.ReadOnlyDirectory)
    ]
    return directories


def get_list_of_not_directories(file):
    """Get the list of directories inside a root file
    Parameters:
        - the file oject from uproot
    Returns:
        - a list of objects (not directory) in the file
    """

    not_directories = [
        key for key in file.keys() if not isinstance(file[key], uproot.reading.ReadOnlyDirectory)
    ]
    return not_directories


def csv_string_to_list(value: str, data_type: type = int) -> list:
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
    return [data_type(x) for x in value.split(",")]


def find_and_capture(text: str, pattern: str):
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
    raise ValueError(f"pattern {pattern} not found in text {text} ")


def manipulate_title(text: str):
    """Replace all the # with \  and add $ $
    Parameters:
        - text (str) of the text to manipulate
    Returns:
        - the manipulated text
    """

    new_string = ""

    for char in text:
        if char == "#":
            new_string += "$\\"

    return new_string


def format_latex(list_str: list):
    """
    Format a string as latex with the radioisotopes formatted
    Parameters
        - list_str : a list of strings
    Returns
        - a list of formatted strings

    Example:
        >>> format_latex(["Pb214])
        >>> [$^{214}$Pb]
    """
    list_new = []
    for text in list_str:
        modified_string = text.replace("Pb214", "$^{214}$Pb")
        modified_string = modified_string.replace("2vbb", "$2\\nu\\beta\\beta$")
        modified_string = modified_string.replace("Co60", "$^{60}$Co")
        modified_string = modified_string.replace("Bi214", "$^{214}$Bi")
        modified_string = modified_string.replace("Tl208", "$^{208}$Tl")
        modified_string = modified_string.replace("K40", "$^{40}$K")
        modified_string = modified_string.replace("K42", "$^{42}$K")
        modified_string = modified_string.replace("Bi212", "$^{212}$Bi")
        modified_string = modified_string.replace("Ac228", "$^{228}$Ac")
        modified_string = modified_string.replace("Ar39", "$^{39}$Ar")

        list_new.append(modified_string)
    return list_new


def ttree2df_all(filename: str, data: str) -> pd.DataFrame:
    """Use uproot to import a TTree and save to a dataframe
    Parameters:
        - filename: file to open
        - tree_name: Which TTree to look at
    Returns:
        - Pandas DataFrame of the data
    """

    file = uproot.open(filename, object_cache=None)

    # Access the TTree inside the file
    tree = None
    for key in file.keys():
        if data in key:
            tree = file[key]
    if tree == None:
        raise ValueError(f"a tree containing {data} not found in the file {filename}")
    # Get the list of branches in the TTree
    branches = tree.keys()

    # Define a dictionary to store the branch data
    data = {}

    # Loop through each branch and extract the data
    cache = 10000
    leng = tree.num_entries
    idx = 0
    list_of_df = []
    tot_length = 0
    for branch_name in branches:
        # Use uproot to read the branch into a NumPy array
        data[branch_name] = tree[branch_name].array()
        df = pd.DataFrame(data)

    return df


def ttree2df(filename: str, data: str, query: str = None, N: int = 500000) -> pd.DataFrame:
    """
    Use uproot to import a TTree and save to a dataframe only reads a subset defined by the query and N
    To read all a dataframe instead use tree2df_all
    Parameters:
        - filename: file to open
        - tree_name: Which TTree to look at
        - query: str of a selection to make (default None)
        - N number of events to read (default: 50000)
    Returns:
        Pandas DataFrame of the data
    """

    file = uproot.open(filename, object_cache=None)

    # Access the TTree inside the file
    tree = None
    for key in file.keys():
        if data in key:
            tree = file[key]
    if tree == None:
        raise ValueError(f"a tree containing {data} not found in the file {filename}")
    # Get the list of branches in the TTree
    branches = tree.keys()

    # Define a dictionary to store the branch data
    data = {}

    # Loop through each branch and extract the data
    cache = 10000
    leng = tree.num_entries
    idx = 0
    list_of_df = []
    tot_length = 0
    while (idx + 1) * cache < leng and tot_length < N:
        for branch_name in branches:
            # Use uproot to read the branch into a NumPy array
            data[branch_name] = tree[branch_name].array(
                entry_start=idx * cache, entry_stop=(idx + 1) * cache
            )
        df = pd.DataFrame(data)

        if query != None:
            df = df.query(query)
        idx += 1
        tot_length += len(df)
        list_of_df.append(df)
    ### read the rest
    for branch_name in branches:
        data[branch_name] = tree[branch_name].array(
            entry_start=idx * cache, entry_stop=(N - tot_length + idx * cache)
        )
    df = pd.DataFrame(data)

    if query != None:
        df = df.query(query)
        list_of_df.append(df)

    # Create a Pandas DataFrame from the dictionary
    return pd.concat(list_of_df, ignore_index=True)


def plot_two_dim(
    varx: np.ndarray,
    vary: np.ndarray,
    rangex: tuple,
    rangey: tuple,
    titlex: str,
    titley: str,
    title: str,
    bins: tuple,
    show=False,
    save="",
    pdf=None,
):
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

    if show == True:
        fig, axes = lps.subplots(1, 1, figsize=(4, 6), sharex=True, gridspec_kw={"hspace": 0})

        h = axes.hist2d(
            varx, vary, bins=bins, cmap="viridis", range=[rangex, rangey], cmin=1, edgecolor="none"
        )

        # fig.colorbar(h[3], ax=axes, label='Counts')
        axes.set_xlabel(titlex)
        axes.set_ylabel(titley)
        axes.set_title(title)
        plt.grid()

    correlation_coefficient = np.corrcoef(varx, vary)[0, 1]

    # Annotate the plot with correlation coefficient
    if show == True:
        axes.annotate(
            f"Correlation = {correlation_coefficient:0.2f}",
            (0.6, 0.88),
            xycoords="axes fraction",
            fontsize=10,
        )
        if pdf == None:
            plt.savefig(save)
        else:
            pdf.savefig()
        plt.close()

    return correlation_coefficient


def plot_correlation_matrix(corr_matrix: np.ndarray, title: str, pdf, show=False):
    """
    Plots a correlation matrix

    Parameters:
    - corr_matrix (numpy.ndarray): 2D NumPy array representing the correlation matrix.
    - title (str): title for the plot
    - pdf: PdfPages object to save the plot
    - show: boolean to show (True) or Save (false) the plot
    """

    # Set up the matplotlib figure
    fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0})

    # Create a heatmap
    cax = axes.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes.set_title(title)

    # Add a colorbar
    cbar = fig.colorbar(cax, shrink=1.0)

    cbar.set_label("$\\rho$")
    plt.grid()
    # Show the plot
    pdf.savefig()

    if show == False:
        plt.close()
    else:
        plt.show()


def plot_table(df, pdf):
    """
    Plot a df as a table
    Parameters:
        -df: pandas dataframe to plot
        -pdf:PdfPages object to save output
    """
    # Create a DataFrame

    # Plot a table
    fig, ax = plt.subplots(figsize=(2, 6 * len(df.values) / 29))  # Adjust the figsize as needed
    ax.axis("off")  # Turn off axis labels

    # Create the table
    table_data = df.T.reset_index().values

    table = ax.table(
        cellText=df.values,
        colLabels=df.keys(),
        cellLoc="center",
        loc="center",
        colWidths=(0.2, 0.8),
    )

    # Style the table
    table.auto_set_font_size(True)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table.set_zorder(100)

    # Show the plot
    pdf.savefig()
    plt.close()


def twoD_slice(matrix, index):
    """
    Function to make a slicing of a matrix
    Parameters:
        - matrix (np.ndarray) of the correaltion matrix
        - index: the indexs to select
    Returns:
        - new sliced np.ndarray
    """
    index = np.array(index)
    matrix_new = matrix[:, index]
    matrix_new = matrix_new[index, :]
    return matrix_new


def get_nth_largest(matrix, n):
    """
    Get the nth largest element in the matrix and its index
    Parameters:
        - matrix: np.ndarray
        - n: the number of elements to extract
    Returns:
        - the values for the n-largest elements
        - the row indices
        - the column indices
    """
    sort_m = matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i >= j:
                sort_m[i, j] = 0

    indices_nth_largest = np.argsort(sort_m.flatten())[-(n + 1)]

    row_indices, col_indices = np.unravel_index(indices_nth_largest, matrix.shape)

    return sort_m[row_indices, col_indices], row_indices, col_indices


def plot_corr(df: pd.DataFrame, i: int, j: int, labels: list, pdf=None):
    """
    Makes a 2D correalation plot
    Parameters:

    """
    key1 = df.keys()[i]
    key2 = df.keys()[j]
    x = np.array(df[key1])
    y = np.array(df[key2])
    rangex = (0, max(x))
    rangey = (0, max(y))
    bins = (100, 100)

    cor = plot_two_dim(
        x,
        y,
        rangex,
        rangey,
        f"{labels[i]} [1/yr]",
        f"{labels[j]} [1/yr]",
        f"{labels[i]} vs {labels[j]}",
        bins,
        True,
        pdf=pdf,
    )


def get_data_numpy(
    type_plot: str, df_tot: pd.DataFrame, name: str, type="marg"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Get the data from the dataframe into some numpy arrays:
    Parameters:
        - type_plot (str): The type of data to extract (parameter,scaling_factor,fit_range or bi_range)
        - df_tot (pd.DataFrame) a pandas dataframe of the data
        - name (str): The type of data (M1,M2,all or a particular spectrum name)
    Returns:
        - a numpy array of the x (indexs)
        - a numpy array of the y data (activity)
        - numpy arrays of y (activity) low/high errors
        - a norm factor for assay measurments
    """
    if type_plot == "parameter":
        x, y, y_low, y_high = get_from_df(df_tot, "fit_range", label=f"_{name}", type=type)
        norm = np.array(df_tot[f"fit_range_orig_{name}"])
        x = x / norm
        y = y / norm
        y_low = y_low / norm
        y_high = y_high / norm
        assay_norm = 1
    elif type_plot == "scaling_factor":
        x, y, y_low, y_high = get_from_df(df_tot, "scaling_factor")
        assay_norm = 1

    elif type_plot == "fit_range" or type_plot == "bi_range":
        x, y, y_low, y_high = get_from_df(df_tot, type_plot, label="_" + name)
        norm = np.array(df_tot[f"{type_plot}_orig_{name}"])
        assay_norm = norm
    else:
        raise ValueError(
            "Error type plot must be 'parameter', 'scaling_factor', 'fit_range' or 'bi_range'"
        )

    return x, y, abs(y_low), abs(y_high), assay_norm


def get_df_results(trees: list, dataset_name: str, specs: list, outfile: str) -> pd.DataFrame:
    """
    Get the fit results into a dataframe merging the different spectrums
    Parameters:
        -trees (list): A list of TTrees in the analysis file (corresponding to spectra in the fit)
        -dataset_name (str): The name of the fitted dataset
        -specs (list): List of spectra for the fit
        -outfile (str):Path to the analysis file
    Returns:
        -pd.DataFrame combining the results

    """
    firstM1 = True
    firstM2 = True
    idx = 0
    for tree in trees:
        ## get spectrum name and multiplicity
        ## ---------------------------------
        index = tree.find(dataset_name)
        print(index)
        print(tree)
        print(specs)
        if index != -1:
            spec_name = tree[index + 1 + len(dataset_name) :].split(";")[0]
            multi_string = [string for string in specs if spec_name in string]
            print(spec_name)
            print(multi_string)
            if len(multi_string) != 1:
                raise ValueError("Error we have multiple spectrum per dataset in the out file")
            multi = multi_string[0].split("/")[0]
        else:
            continue

        df = pd.DataFrame(ttree2df_all(outfile, tree))
        df = df[df["fit_range_orig"] != 0]

        ### add total range (M1 and M2)
        ### ------------------------------------------
        for key in df.keys():
            if "range" in key:
                df[key + "_all_spec"] = df[key]

        if multi == "mul_surv":
            for key in df.keys():
                if ("range" in key) and ("all" not in key):
                    df[key + "_M1"] = df[key]
            firstM1 = False
        elif "mul2" in multi:
            for key in df.keys():
                if ("range" in key) and ("all" not in key):
                    df[key + "_M2"] = df[key]
            firstM2 = False

        for key in df.keys():
            if ("range" in key) and ("M2" not in key) and ("M1" not in key) and ("all" not in key):
                df.rename(columns={key: key + "_" + spec_name}, inplace=True)

        ### merge dataframes

        if idx == 0:
            df_tot = df
        else:
            ### append info in an appropriate
            df_tot = merge_dfs(df_tot, df)
        idx = idx + 1

    return df_tot


def make_error_bar_plot(
    indexs,
    labels: list,
    y: np.ndarray,
    ylow: np.ndarray,
    yhigh: np.ndarray,
    data="all",
    name_out=None,
    obj="parameter",
    y2=None,
    ylow2=None,
    yhigh2=None,
    label1=None,
    label2=None,
    extra=None,
    do_comp=0,
    low=0.1,
    scale="log",
    upper=0,
    data_band=None,
    categories=None,
    split_priors=False,
    has_prior=None,
    assay_mean=None,
    assay_high=None,
    assay_low=None,
):
    """
    Make the error bar plot
    """
    indexs = np.array(indexs, dtype="int")
    vset = tc.tol_cset("vibrant")
    labels = np.array(labels)
    y = y[indexs]
    labels = labels[indexs]
    ylow = ylow[indexs]
    yhigh = yhigh[indexs]

    if assay_mean is not None:
        assay_mean = assay_mean[indexs]
        assay_high = assay_high[indexs]
        assay_low = assay_low[indexs]

    if split_priors == True:
        has_prior = np.array(has_prior, dtype="bool")[indexs]

        index_prior = np.where(has_prior)[0]
        index_no_prior = np.where(~has_prior)[0]

    if do_comp == True:
        ylow2 = ylow2[indexs]
        yhigh2 = yhigh2[indexs]
        y2 = y2[indexs]

    ### get the indexs with prior and those without priors
    ### ------------------------------------------------------------
    xin = np.arange(len(labels))

    height = 1 + len(y) * 0.32
    lps.use("legend")
    fig, axes = lps.subplots(1, 1, figsize=(4.5, 4), sharex=True, gridspec_kw={"hspace": 0})

    ### get the priors
    ### ----------------------------------------------

    if split_priors == True and has_prior is None:
        raise ValueError("Splitting by those component with priors requires to set 'has_prior'")

    ### shorten labels
    for i in range(len(labels)):
        label = labels[i]

        ### shorten
        if "hpge_support_copper" in label:
            labels[i] = labels[i].split("hpge")[0] + "hpge_copper"
        if "front" in label:
            labels[i] = labels[i].split("front")[0] + "fe_electr"
        if "hpge_in" in label:
            labels[i] = labels[i].split("hpge")[0] + "insulators"

    ### split into contributions with and without priors

    if split_priors == True:
        if do_comp == True:
            raise NotImplementedError(
                "It is not implemented to both split priors and compare 2 fits"
            )

        yo = np.array(y)
        ylowo = np.array(ylow)
        yhigho = np.array(yhigh)

        y = yo[index_prior]
        ylow = ylowo[index_prior]
        yhigh = yhigho[index_prior]
        y2 = yo[index_no_prior]
        ylow2 = ylowo[index_no_prior]
        yhigh2 = yhigho[index_no_prior]
        xin1 = xin[index_prior]
        xin2 = xin[index_no_prior]

    ### either split by category or not
    if categories is None:
        if assay_mean is not None:
            axes.errorbar(
                y=xin,
                x=assay_mean,
                fmt="o",
                color="black",
                ecolor="grey",
                markersize=1,
                label="Radioassy",
                alpha=0.4,
            )
            for i in range(len(xin)):
                axes.fill_betweenx(
                    [xin[i] - 0.2, xin[i] + 0.2],
                    assay_mean[i] - assay_low[i],
                    assay_mean[i] + assay_high[i],
                    alpha=0.4,
                    color="gray",
                    linewidth=0,
                )

        if do_comp == False and split_priors == False:
            axes.errorbar(
                y=xin,
                x=y,
                xerr=[abs(ylow), abs(yhigh)],
                fmt="o",
                color=vset.blue,
                ecolor=vset.cyan,
                markersize=1,
                label="MC",
            )
            # a=1
        elif split_priors == True:
            axes.errorbar(
                y=xin1,
                x=y,
                xerr=[abs(ylow), yhigh],
                fmt="o",
                color=vset.blue,
                ecolor=vset.cyan,
                markersize=1,
                label="With prior",
            )
            axes.errorbar(
                y=xin2,
                x=y2,
                xerr=[abs(ylow2), abs(yhigh2)],
                fmt="o",
                color=vset.red,
                ecolor=vset.orange,
                markersize=1,
                label="Without prior",
            )
        else:
            axes.errorbar(
                y=xin + 0.15 * len(y) / 30,
                x=y,
                xerr=[abs(ylow), abs(yhigh)],
                fmt="o",
                color=vset.blue,
                ecolor=vset.cyan,
                markersize=1,
                label=label1,
            )
            axes.errorbar(
                y=xin - 0.15 * len(y) / 30,
                x=y2,
                xerr=[abs(ylow2), abs(yhigh2)],
                fmt="o",
                color=vset.orange,
                ecolor=vset.magenta,
                markersize=1,
                label=label2,
            )

    else:
        if do_comp == True:
            raise ValueError("Splitting by category not implemented for comparison")

        cat_sort = np.argsort(categories)

        labels = labels[cat_sort]
        categories = categories[cat_sort]
        y = y[cat_sort]
        ylow = ylow[cat_sort]
        yhigh = yhigh[cat_sort]
        colors = [vset.blue, vset.teal, vset.magenta, vset.cyan]
        for cat, col in zip(sorted(set(categories)), colors):
            y_tmp = y[categories == cat]
            ylow_tmp = ylow[categories == cat]
            yhigh_tmp = yhigh[categories == cat]
            xin_tmp = xin[categories == cat]
            axes.errorbar(
                y=xin_tmp,
                x=y_tmp,
                xerr=[ylow_tmp, yhigh_tmp],
                fmt="o",
                color=col,
                ecolor=col,
                markersize=1,
                label=cat,
            )

    if data_band is not None:
        axes.axvline(x=data_band[0], color="red", linestyle="--", label="Data")
        axes.axvspan(
            xmin=data_band[0] - data_band[1],
            xmax=data_band[0] + data_band[2],
            color=vset.orange,
            alpha=0.3,
        )

    ### set upper limits for plots
    if upper == 0:
        if scale == "linear":
            if len(y) > 0:
                upper = np.max(y + 1.5 * yhigh)
        elif len(y) > 0:
            upper = np.max(y + 1.5 * yhigh)
        if do_comp == True or split_priors == True:
            upper = max(upper, np.max(y2 + 1.5 * yhigh2))

        if scale == "log":
            upper *= 3

    axes.set_yticks(xin, labels)

    ## draw data band
    if data_band is not None:
        upper = data_band[0] + 2 * data_band[2]

    print(data_band)
    print(upper)

    ### set the labels
    ### -------------------------
    if obj == "fit_range":
        axes.set_xlabel(f"Recon. counts / yr in {data} data")
        axes.set_xlim(1, upper)

    elif obj == "bi_range":
        axes.set_xlabel(f"Recon. bkg counts / yr in {data} data")
        axes.set_xlim(low, upper)
    elif obj == "scaling_factor":
        axes.set_xlabel("Scaling factor [1/yr] ")
        axes.set_xlim(1e-7, upper)
    elif obj == "parameter":
        axes.set_xlabel("Activity [$\mu$Bq]")
        axes.set_xlim(low, upper)
    elif obj == "frac":
        axes.set_xlabel(f"Frac. of counts in {data} [%]")
        axes.set_xlim(1, upper)

    else:
        axes.set_xlabel(f"Recon. counts / yr in {obj} data")
        axes.set_xlim(1, upper)

    axes.set_yticks(axes.get_yticks())
    fonti = 11
    length = len(y)
    if y2 is not None:
        length += len(y2)

    if length > 15 and do_comp == False:
        fonti = 4
    axes.set_yticklabels([val for val in labels], fontsize=fonti)
    axes.set_xscale(scale)

    plt.grid()
    if do_comp == True or split_priors == True or data_band is not None or categories is not None:
        leg = axes.legend(
            loc="best", edgecolor="black", frameon=True, facecolor="white", framealpha=1
        )
        leg.set_zorder(10)


def replace_e_notation(my_string):
    # Using regular expression to match e+0n where n is any character
    pattern = re.compile(r"e\+0(.)")
    modified_string = re.sub(pattern, r"\\cdot 10^{\1}", my_string)
    return modified_string


def priors2table(priors: dict):
    """Convert the priors json into a latex table"""
    print(json.dumps(priors, indent=1))

    convert_fact = 1 / 31.5
    print(
        "\multicolumn{1}{c}{\cellcolor[HTML]{CBCEFB}\\textbf{Source} &\multicolumn{1}{c}{\cellcolor[HTML]{CBCEFB} \\textbf{} Decay} & \multicolumn{1}{c}{\cellcolor[HTML]{CBCEFB} \\textbf{Activity [$\mu$Bq]} & \multicolumn{1}{c}{\cellcolor[HTML]{CBCEFB}\\textbf{Type} \\\\ \hline \hline "
    )
    first_Bi = 0
    first_Tl = 0
    for comp in priors["components"]:
        source = comp["full-name"]
        if "Bi212Tl208" in comp["name"] and first_Tl == 0:
            decay = "$^{212}$Bi+$^{208}$Tl"
            first_Tl = 1
        elif "Pb214Bi214" in comp["name"] and first_Bi == 0:
            decay = "$^{214}$Pb+$^{214}$Bi"
            first_Bi = 1
        else:
            decay = ""
        type_meas = comp["type"]
        if type_meas == "icpms":
            type_meas = "ICP-MS"
        if type_meas == "guess":
            type_meas = "Guess"
        if type_meas == "hpge":
            type_meas = "HPGe"
        prior = comp["prior"]
        if "gaus" in prior:
            split_values = prior.split(":")

            param1, param2, param3 = map(float, split_values[1].split(","))
            a = -param2 / param3

            rv = truncnorm(a=a, b=5, loc=param2, scale=param3)
            high = param2 + 5 * param3
            low_err = 0 if param3 > param2 else param2 - param3
            high_err = param3 + param2
            best = param2
            low_err *= convert_fact
            high_err = convert_fact * param3
            best *= convert_fact
            meas = f"${best:.2g} \pm{high_err:.2g}$"
            meas = replace_e_notation(meas)
        elif "exp" in prior:
            split_parts = prior.split("/")

            # Extract the upper limit from the exponenital
            if len(split_parts) > 1:
                upper_limit = float(split_parts[1][:-1])
                rv = expon(scale=upper_limit / 2.3)
                high = 2 * upper_limit
                low_err = 0
                high_err = upper_limit
                high_err *= convert_fact
                meas = f"$<{high_err:.2g}$"
                meas = replace_e_notation(meas)

        print(f"{source} & {decay} & {meas} & {type_meas} \\\\")


def get_index_by_type(names):
    """Get the index for each type (U,Th,K)"""
    i = 0
    index_U = []
    index_Th = []
    index_K = []
    index_2nu = []

    for key in names:
        if key.find("Bi212") != -1:
            index_Th.append(i)
        elif key.find("Bi214") != -1:
            index_U.append(i)
        elif key.find("K42") != -1:
            index_K.append(i)
        elif key.find("2v") != -1:
            index_2nu.append(i)
        i = i + 1

    return {"U": index_U, "2nu": index_2nu, "Th": index_Th, "K": index_K}


def get_from_df(df, obj, label="", type="marg"):
    """Get the index, y and errors from dataframe"""

    x = np.array(df.index)
    if f"{obj}_{type}_mod{label}" in df:
        mode = "mod"
    else:
        mode = "mode"
    y = np.array(df[f"{obj}_{type}_{mode}{label}"])
    y_low = y - np.array(df[f"{obj}_qt16{label}"])
    y_high = np.array(df[f"{obj}_qt84{label}"]) - y
    for i in range(len(y_low)):
        if y_low[i] < 0:
            y_low[i] = 0
            y_high[i] += y[i]
            y[i] = 0

    return x, y, y_low, y_high


def get_error_bar(N: float):
    """
    A poisson error-bar for N observed counts.
    """

    x = np.linspace(0, 5 + 2 * N, 5000)
    y = poisson.pmf(N, x)
    integral = y[np.argmax(y)]
    bin_id_l = np.argmax(y)
    bin_id_u = np.argmax(y)

    integral_tot = np.sum(y)
    while integral < 0.683 * integral_tot:
        ### get left bin
        if bin_id_l > 0 and bin_id_l < len(y):
            c_l = y[bin_id_l - 1]
        else:
            c_l = 0

        if bin_id_u > 0 and bin_id_u < len(y):
            c_u = y[bin_id_u + 1]
        else:
            c_u = 0

        if c_l > c_u:
            integral += c_l
            bin_id_l -= 1
        else:
            integral += c_u
            bin_id_u += 1

    low_quant = x[bin_id_l]
    high_quant = x[bin_id_u]
    return N - low_quant, high_quant - N


def integrate_hist(hist, low, high):
    """Integrate the histogram"""

    bin_centers = hist.axes.centers[0]

    values = hist.values()
    lower_index = np.searchsorted(bin_centers, low, side="right")
    upper_index = np.searchsorted(bin_centers, high, side="left")
    bin_contents_range = values[lower_index:upper_index]
    bin_centers_range = bin_centers[lower_index:upper_index]

    return np.sum(bin_contents_range)


def get_total_efficiency(det_types, cfg, spectrum, regions, pdf_path, det_sel="all", mc_list=None):
    eff_total = {}
    ### creat the efficiency maps (per dataset)
    for det_name, det_info in det_types.items():
        det_list = det_info["names"]
        effs = {}
        for key in regions:
            effs[key] = {}

        for det, named in zip(det_list, det_info["types"]):
            eff_new, good = get_efficiencies(
                cfg, spectrum, det, regions, pdf_path, named, "mul_surv", mc_list=mc_list
            )
            if good == 1 and (named == det_sel or det_sel == "all"):
                effs = sum_effs(effs, eff_new)

        eff_total[det_name] = effs

    return eff_total


def get_efficiencies(
    cfg,
    spectrum,
    det_type,
    regions,
    pdf_path,
    name,
    spectrum_fit="",
    mc_list=None,
    type_fit="icpc",
    idx=0,
):
    """Get the efficiencies or the fraction of events in a given spectrum depositing energy in each region.
    Parameters:
        -cfg (dict): the fit configuration file
        - spectrum (str): name of the spectrum
        - det_type (str): detector type to look at
        - regions (dict): dictonary of regions to look at
        - pdf_path (str): path to the pdf files
        - spectrum_fit (str): the spectrum used for the fit (to get the list of components)
        - mc_list (list): list of MC files to consider
        - type_fit (str): used to extract list of MC files
        - idx (int ) : index of the datafile to look into
    Returns
        dict of the efficiencies

    """

    if det_type in ["icpc", "ppc", "coax", "bege"] and type_fit != "all":
        type_fit = det_type

    effs = {}
    for key in regions:
        effs[key] = {}

    if spectrum_fit == "":
        spectrum_fit = spectrum

    if mc_list is None:
        for key, region in regions.items():
            if type_fit != "all":
                effs[key]["2vbb_bege"] = 0
                effs[key]["2vbb_coax"] = 0
                effs[key]["2vbb_ppc"] = 0
                effs[key]["2vbb_icpc"] = 0
                effs[key]["K42_hpge_surface_bege"] = 0
                effs[key]["K42_hpge_surface_coax"] = 0
                effs[key]["K42_hpge_surface_ppc"] = 0
                effs[key]["K42_hpge_surface_icpc"] = 0
                effs[key]["alpha_ppc"] = 0
                effs[key]["alpha_bege"] = 0
                effs[key]["alpha_coax"] = 0
                effs[key]["alpha_icpc"] = 0
        for key in cfg["fit"]["theoretical-expectations"].keys():
            if ".root" in key:
                filename = key
        filename = list(cfg["fit"]["theoretical-expectations"].keys())[idx]
        print(filename)
        if f"{spectrum_fit}/{type_fit}" in cfg["fit"]["theoretical-expectations"][filename]:
            icpc_comp_list = cfg["fit"]["theoretical-expectations"][filename][
                f"{spectrum_fit}/{type_fit}"
            ]["components"]
        else:
            warnings.warn(f"{spectrum_fit}/{det_type} not in MC PDFs")
            return effs, 0
    else:
        icpc_comp_list = mc_list

    comp_list = copy.deepcopy(icpc_comp_list)
    for comp in comp_list:
        for key in comp["components"].keys():
            par = key
        ## now open the file

        if "root-file" in comp.keys():
            with uproot.open(pdf_path + comp["root-file"], object_cache=None) as file:
                if f"{spectrum}/{det_type}" in file:
                    hist = file[f"{spectrum}/{det_type}"]
                    N = int(file["number_of_primaries"])
                    hist = hist.to_hist()
                    for key, region in regions.items():
                        eff = 0
                        for region_tmp in region:
                            eff += float(integrate_hist(hist, region_tmp[0], region_tmp[1]))
                        effs[key][par] = eff / N
                else:
                    warnings.warn(f"{spectrum}/{det_type} not in MC PDFs")
                    for key, region in regions.items():
                        effs[key][par] = 0

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
            for key, region in regions.items():
                eff = 0
                for region_tmp in region:
                    eff += (
                        p0 * (region_tmp[1] - region_tmp[0])
                        + p1 * (region_tmp[1] * region_tmp[1] - region_tmp[0] * region_tmp[0]) / 2
                    )
                effs[key][par] = eff

                if det_type not in ["icpc", "bege", "ppc", "coax"]:
                    effs[key][par] = 0

    return effs, 1


def sum_effs(eff1: dict, eff2: dict) -> dict:
    """
    Function to sum the two efficiency dictonaries up to two layers

    Parameters:
        - eff1: (dict) the first dictonary
        - eff2: (dict) the second
    Returns:
        - dictonary where every item with the same key is summed
    """

    dict_sum = {}
    for key in set(eff1) | set(eff2):  # Union of keys from both dictionaries
        ## sum two layers
        dict_sum[key] = {}

        if isinstance(eff1[key], dict) or isinstance(eff2[key], dict):
            for key2 in set(eff1[key]) | set(eff2[key]):
                dict_sum[key][key2] = eff1[key].get(key2, 0) + eff2[key].get(key2, 0)

        ## sum one layer
        else:
            dict_sum[key] = eff1.get(key, 0) + eff2.get(key, 0)
    return dict_sum


def get_data_counts_total(
    spectrum: str, det_types: dict, regions: dict, file: str, det_sel: str = None, key_list=[]
):
    """
    Function to get the data counts in different regions
    Parameters:
        - spectrum:  the data spectrum to use (str)
        - det_types: the dictonary of detector types
        - regions:   the dictonary of regions
        - file   :   str path to the file
        - det_sel : a particular detector type (def None)
    Returns:
        - a dictonary of the counts in each region

    """
    data_counts_total = {}
    for det_name, det_info in det_types.items():
        det_list = det_info["names"]
        dt = det_info["types"]
        data_counts = {}
        for key in key_list:
            data_counts[key] = 0

        for det, type in zip(det_list, dt):
            if type == det_sel or det_sel == None:
                data_counts = sum_effs(data_counts, get_data_counts(spectrum, det, regions, file))

        data_counts_total[det_name] = data_counts

    return data_counts_total


def create_efficiency_likelihood(data_surv: np.ndarray, data_cut: np.ndarray):
    """
    A method to create the efficiency likelihood,
    This makes the likelihood function to mimimize based on a list of data_surving and being cut (counts in each bin)
    Parameters:
        - data_surv: np.ndarray of the counts in each bin survivin
        - data_cut: np.ndarray of the counts in each bin being cut
    Returns:
        - the likelihood as python exectuable
    """

    def likelihood(Ns: float, Nb: float, eff_s: float, eff_b: float):
        """
        The likelihood function itself, based on 4 parameters
        This is a sum over the likelihood for the cut and surviving spectrum
        Parameters:
        - Ns the number of signal counts
        - Nb the number of bkg
        - eff_s the efficiency for signal
        - eff_b the efficiency for background
        Returns:
        - negative log likelihood P(D|theta)
        """

        logL = 0
        preds_surv = np.array([Nb * eff_b, Nb * eff_b + Ns * eff_s, Nb * eff_b])
        preds_cut = np.array(
            [Nb * (1 - eff_b), Nb * (1 - eff_b) + Ns * (1 - eff_s), Nb * (1 - eff_b)]
        )

        logL += sum(poisson.logpmf(data_surv, preds_surv))
        logL += sum(poisson.logpmf(data_cut, preds_cut))

        return -logL

    return likelihood


def create_graph_likelihood(func, x, y, el, eh):
    """
    Method to create the likelihood to fit a graph

    """

    def likelihood(*pars):
        logL = 0

        logLs = (func(x, *pars) > y) * (-np.power((func(x, *pars) - y), 2) / (2 * eh))
        logLs += (func(x, *pars) <= y) * (-np.power((func(x, *pars) - y), 2) / (2 * el))
        logL = sum(logLs)

        return -logL

    return likelihood


def create_counting_likelihood(data, bins):
    """Create the likelihood function"""

    def likelihood(Ns, Nb):
        """The likelihood function"""

        logL = 0
        preds_surv = np.array([Nb * bins[0] / bins[1], Nb + Ns, Nb * bins[2] / bins[1]])

        logL += sum(poisson.logpmf(data, preds_surv))

        return -logL

    return likelihood


def plot_counting_calc(energies, data, values):
    """Make a plot to show the efficiency calculation"""

    vset = tc.tol_cset("vibrant")
    Nb = values["Nb"]
    Ns = values["Ns"]

    bkg = np.array([Nb, Nb, Nb, Nb])
    sig = np.array([0, Ns, 0])

    fig, axes = lps.subplots(1, 1, figsize=(4, 4), sharex=True, gridspec_kw={"hspace": 0})
    centers = [(energies[i] + energies[i + 1]) / 2 for i in range(len(energies) - 1)]

    axes.bar(
        centers,
        data,
        width=np.diff(energies),
        edgecolor=vset.blue,
        align="center",
        color=vset.blue,
        alpha=0.3,
        linewidth=0,
        label="Data",
    )
    axes.step(energies, bkg, where="mid", color=vset.red, alpha=1, linewidth=1, label="Bkg.")
    axes.step(centers, sig, where="mid", color=vset.teal, alpha=1, linewidth=1, label="Sig")

    axes.legend(loc="best")

    axes.set_xlim(energies[0], energies[-1])

    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("Counts/bin")

    axes.set_ylabel("Counts/bin")

    axes.set_title(f"Events for {centers[1]} keV", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"plots/relative_intensity/counts_calc_{centers[1]}.pdf")
    plt.close()


def plot_relative_intensities_triptych(
    outputs, data, data_err, orders, title, savepath, no_labels=False
):
    """
    Make the triple plot for relative intesntiies
    """

    vset = tc.tol_cset("vibrant")

    ratios = []

    names = []
    for i in range(len(outputs)):
        ratios_tmps = []
        for component in orders:
            ratio_tmp = outputs[i][component]
            ratios_tmps.append(ratio_tmp)

            if i == 0:
                if (component != "front_end_electronics") and (component != "hpge_support_copper"):
                    names.append(component)
                elif component == "front_end_electronics":
                    names.append("fe_electronics")
                else:
                    names.append("hpge_copper")
        ratios_tmps = np.array(ratios_tmps)
        ratios.append(ratios_tmps)
    names = np.array(names)

    fig, axes = lps.subplots(
        1, 3, figsize=(6, 3.5), sharex=False, sharey=True, gridspec_kw={"hspace": 0}
    )

    for i in range(len(outputs)):
        axes[i].set_xlim(0, 1.2 * max(max(ratios[i]), data[i] + data_err[i]))
        axes[i].set_title(title[i], fontsize=16)
        axes[i].errorbar(ratios[i], names, color=vset.blue, fmt="o", label="MC")
        if i == 1:
            axes[i].set_xlabel("Relative Intensity [%]")
        axes[i].axvline(x=data[i], color=vset.red, label="data")
        axes[i].axvspan(
            xmin=data[i] - data_err[i], xmax=data[i] + data_err[i], alpha=0.2, color=vset.orange
        )
        if i == 0:
            axes[i].set_yticks(np.arange(len(names)), names, rotation=0, fontsize=11)
        if i == 2:
            axes[i].legend(
                loc="best", edgecolor="black", frameon=True, facecolor="white", framealpha=1
            )

    plt.tight_layout()
    plt.savefig(savepath)


def plot_relative_intensities(outputs, data, data_err, orders, title, savepath, no_labels=False):
    ### now loop over the keys

    vset = tc.tol_cset("vibrant")

    ratios = []
    errors = []
    names = []
    for component in orders:
        ratio_tmp = outputs[component]
        rt = ratio_tmp
        if isinstance(rt, tuple):
            ratio_tmp = rt[0]
            error = rt[1]
        else:
            ratio_tmp = rt
            error = 0
        ratios.append(ratio_tmp)

        names.append(component)
        errors.append(error)
    ratios = np.array(ratios)
    names = np.array(names)

    fig, axes = lps.subplots(1, 1, figsize=(3, 3), sharex=True, gridspec_kw={"hspace": 0})
    axes.set_xlim(0, 1.3 * max(max(ratios), data + data_err))
    axes.set_title(title, fontsize=14)

    axes.errorbar(ratios, names, xerr=[errors, errors], color=vset.blue, fmt="o", label="MC")
    axes.set_xlabel("Relative Intensity [%]")
    axes.set_yticks(np.arange(len(names)), names, rotation=0, fontsize=7)
    axes.axvline(x=data, color=vset.red, label="data")
    axes.axvspan(xmin=data - data_err, xmax=data + data_err, alpha=0.2, color=vset.orange)
    axes.legend(loc="best", edgecolor="black", frameon=True, facecolor="white", framealpha=1)
    plt.tight_layout()
    plt.savefig(savepath)


def normalized_poisson_residual(mu:np.ndarray, obs:np.ndarray)->np.ndarray:
    """ Compute normalised poisson residuals.

    For mu > 50 a Gaussian approximation is employed. When mu is less
    than 50, the tail probability is calculated integrating from the mode,
    this is used to calculate a p-value then converted into a residual.

    Parameters
    ----------
    mu   
        The model predictions.
    obs
        The observed data.
    """

    if obs < 1:
        obs = 0
    if mu < 50:
        ## get mode

        mode = math.floor(mu)

        if obs == 0 and mode == 0:
            return 0
        if obs < mode:
            prob = poisson.cdf(obs, mu, loc=0)
            sign = -1
        else:
            prob = 1 - poisson.cdf(obs - 1, mu, loc=0)
            sign = 1
        return sign * norm.ppf(1 - prob)
    return (obs - mu) / np.sqrt(mu)


def plot_eff_calc(energies, data_surv, data_cut, values):
    """Make a plot to show the efficiency calculation"""

    vset = tc.tol_cset("vibrant")
    Nb = values["Nb"]
    Ns = values["Ns"]
    eff_s = values["eff_s"]
    eff_b = values["eff_b"]
    bkg = np.array([Nb * eff_b, Nb * eff_b, Nb * eff_b, Nb * eff_b])
    sig = np.array([0, Ns * eff_s, 0])

    bkg_cut = np.array([Nb * (1 - eff_b), Nb * (1 - eff_b), Nb * (1 - eff_b), Nb * (1 - eff_b)])
    sig_cut = np.array([0, Ns * (1 - eff_s), 0])

    fig, axes = lps.subplots(2, 1, figsize=(4, 4), sharex=True, gridspec_kw={"hspace": 0})
    centers = [(energies[i] + energies[i + 1]) / 2 for i in range(len(energies) - 1)]

    axes[0].bar(
        centers,
        data_surv,
        width=np.diff(energies),
        edgecolor=vset.blue,
        align="center",
        color=vset.blue,
        alpha=0.3,
        linewidth=0,
        label="Data",
    )
    axes[0].step(energies, bkg, where="mid", color=vset.red, alpha=1, linewidth=1, label="Bkg.")
    axes[0].step(centers, sig, where="mid", color=vset.teal, alpha=1, linewidth=1, label="Sig")

    axes[0].legend(loc="best")
    axes[1].bar(
        centers,
        data_cut,
        width=np.diff(energies),
        edgecolor=vset.blue,
        align="center",
        color=vset.blue,
        alpha=0.3,
        linewidth=0,
    )
    axes[1].step(energies, bkg_cut, where="mid", color=vset.red, alpha=1, linewidth=1)
    axes[1].step(centers, sig_cut, where="mid", color=vset.teal, alpha=1, linewidth=1)

    axes[1].set_xlim(energies[0], energies[-1])
    axes[0].set_xlim(energies[0], energies[-1])

    axes[0].set_xlabel("Energy [keV]")
    axes[0].set_ylabel("Counts/bin")

    axes[1].set_xlabel("Energy [keV]")

    axes[1].set_ylabel("Counts/bin")
    axes[0].set_ylabel("Counts/bin")

    axes[1].xaxis.set_tick_params(top=False)

    axes[0].set_title(f"Events passing LAr (top) and cut (bottom) for {centers[1]} keV", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"plots/lar/eff_calc_{centers[1]}.pdf")


def extract_prior(prior: str):
    """
    Extracts the prior as a sccipy object (only works for Gauss or Expon)
    Parameters
    -------------------
        - a string of the prior as used in hmixfit
    Returns:
    -------------------
        - scipy random variable of the probability distribution
        - an upper limit on the parameter (5 sigma) used more for plotting
        - the lower error on the activity (sigma)
        - the upper error on the activity (sigma)
    """

    # gaussian prior
    rv = None
    if "gaus" in prior:
        split_values = prior.split(":")

        param1, param2, param3 = map(float, split_values[1].split(","))
        a = -param2 / param3

        rv = truncnorm(a=a, b=5, loc=param2, scale=param3)
        high = param2 + 5 * param3
        low_err = 0 if param3 > param2 else param2 - param3
        high_err = param3 + param2

    elif "exp" in prior:
        split_parts = prior.split("/")

        # Extract the upper limit from the exponenital
        if len(split_parts) > 1:
            upper_limit = float(split_parts[1][:-1])
            rv = expon(scale=upper_limit / 2.3)
            high = 2 * upper_limit
            low_err = 0
            high_err = upper_limit / 1.65
        else:
            raise ValueError("exp prior doesnt contain the part /UpperLimit) needed")
    else:
        raise ValueError("Only supported priors are gaus and exp currently")

    return rv, high, (low_err, high_err)


def merge_dfs(df1, df2):
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    missing_columns_df1 = df2.columns.difference(df1.columns)
    missing_columns_df2 = df1.columns.difference(df2.columns)

    ### add appropriate columns
    for col in missing_columns_df1:
        df1[col] = 0

    for col in missing_columns_df2:
        df2[col] = 0
    ### now handle different rows

    unique_keys_df1 = set(df1["comp_name"])
    unique_keys_df2 = set(df2["comp_name"])

    missing_rows_df1 = df2[df2["comp_name"].isin(unique_keys_df2 - unique_keys_df1)]
    missing_rows_df2 = df1[df1["comp_name"].isin(unique_keys_df1 - unique_keys_df2)]

    for index, row in missing_rows_df1.iterrows():
        columns_to_zero = [col for col in row.index if "range" in col]
        row[columns_to_zero] = 0
        df1 = pd.concat([df1, pd.DataFrame([row])], ignore_index=True)

    for index, row in missing_rows_df2.iterrows():
        columns_to_zero = [col for col in row.index if "range" in col]
        row[columns_to_zero] = 0

        df2 = pd.concat([df2, pd.DataFrame([row])], ignore_index=True)

    result = pd.DataFrame(columns=df1.columns)
    df2 = df2.sort_values(by="comp_name").reset_index(drop=True)
    df1 = df1.sort_values(by="comp_name").reset_index(drop=True)

    for col in df1.columns:
        # Check if the column contains "range"
        if "range" in col:
            # Add corresponding values from both DataFrames
            result[col] = df1[col] + df2[col]
        else:
            # Check if values are the same and take value from the first DataFrame
            if not (df1[col] == df2[col]).all() and not np.allclose(
                df1[col].to_numpy(), df2[col].to_numpy()
            ):
                raise ValueError(
                    f"Values in column '{col}' '{df1[col]}' '{df2[col]}' do not match between the two DataFrames."
                )
            result[col] = df1[col]

    return result


def parse_cfg(cfg_dict: dict, replace_dash: bool = True) -> tuple[str, str, str, list, list, str]:
    """
    Extract a bunch of information from the hmixfit cfg files
    Parameters:
        - cfg_dict: dictonary (cfg file for hmixfit)
        - replace dash: bool to replace dashes in the datset name
    Returns
        - the fit name
        - the output directory of fit results
        - the list of detector types,
        - the ranges for each fit
        - the name of the dataset
        - ds_names
    """

    fit_name = cfg_dict["id"]
    out_dir = "../hmixfit/" + cfg_dict["output-dir"]
    det_types = []
    ranges = []
    ds = []
    dataset_names = []
    os.makedirs(f"plots/summary/{fit_name}", exist_ok=True)

    for key in cfg_dict["fit"]["theoretical-expectations"]:
        for spec in cfg_dict["fit"]["theoretical-expectations"][key]:
            det_types.append(spec)
            if "fit-range" in cfg_dict["fit"]["theoretical-expectations"][key][spec]:
                r = cfg_dict["fit"]["theoretical-expectations"][key][spec]["fit-range"]
                if len(np.shape(r)) == 2:
                    ranges.append([r[0][0], r[1][1]])
                else:
                    ranges.append(
                        cfg_dict["fit"]["theoretical-expectations"][key][spec]["fit-range"]
                    )
            elif "fit-range-y" in cfg_dict["fit"]["theoretical-expectations"][key][spec]:
                ranges.append(cfg_dict["fit"]["theoretical-expectations"][key][spec]["fit-range-y"])

        dataset_names.append(key[:-5])
        ds.append(key)
        if replace_dash:
            dataset_names[-1] = dataset_names[-1].replace("-", "_").replace(".", "_")

    return fit_name, out_dir, det_types, ranges, dataset_names, ds


def plot_mc_no_save(axes, pdf, name="", linewidth=0.6, range=(0, 4000), color="blue"):
    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("counts/10 keV")

    axes.set_yscale("linear")
    axes.set_xlim(range)
    pdf.plot(
        ax=axes,
        linewidth=linewidth,
        yerr=False,
        flow=None,
        histtype="step",
        label=name,
        color=color,
    )
    # pdf.plot(ax=axes, linewidth=0.8, yerr=False,flow= None,histtype="fill",alpha=0.15)

    return axes


def format(title):
    title = title.replace("M1", "$\\mathtt{M1} $")
    title = title.replace("M2", "$\\mathtt{M2} $")
    title = title.replace("coax", " COAX")
    title = title.replace("bege", " BeGe")
    title = title.replace("ppc", " PPC")
    title = title.replace("icpc", " ICPC")

    return title


def compare_mc(
    max_energy, pdfs, decay, order, norm_peak, pdf, xlow, xhigh, scale, linewidth=0.6, colors=[]
):
    """Compare MC files"""
    fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0})
    max_c = 0
    axes.set_xlim(xlow, xhigh)

    for idx, name in enumerate(order):
        Peak_counts = pdfs[decay + "_" + name][norm_peak]

        pdf_norm = scale_hist(pdfs[decay + "_" + name], 1 / Peak_counts)

        max_c = max(pdf_norm[max_energy], max_c)

        axes = plot_mc_no_save(axes, pdf_norm, name=name, linewidth=linewidth, color=colors[idx])

    axes.legend(
        loc="best", edgecolor="black", frameon=True, facecolor="white", framealpha=1, fontsize=8
    )
    axes.set_xlim(xlow, xhigh)
    axes.set_yscale(scale)
    if scale != "log":
        axes.set_ylim(0, 1.1 * max_c)

    pdf.savefig()


def compare_intensity_curves(intensity_map, order, save=""):
    """Compare intensity curves"""

    fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0})

    maxI = 0
    for name in order:
        map_tmp = intensity_map[name]
        E = []
        I = []
        for peak in map_tmp:
            E.append(peak)
            I.append(map_tmp[peak])

        E = np.array(E)
        I = np.array(I) * 100

        axes.plot(E, I, label=name, linewidth=0.6)
        axes.set_xlim(min(E) - 100, max(E) + 100)
        maxI = max(max(I), maxI)

        axes.set_ylim(0, maxI + 30)

        axes.set_xlabel("Peak Energy [keV]")
        axes.set_ylabel("Relative Intensitiy [%]")
    axes.legend(
        loc="upper right",
        edgecolor="black",
        frameon=True,
        facecolor="white",
        framealpha=1,
        fontsize=8,
    )

    plt.savefig(save)


def plot_N_Mc(pdfs, labels, name, save_pdf, data, range_x, range_y, scale, colors, bin=10):
    """Plot the MC files"""

    vset = tc.tol_cset("vibrant")

    fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0})
    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("counts/10 keV")
    axes.set_title(name)
    axes.set_yscale(scale)
    axes.set_xlim(range_x)
    axes.set_ylim(range_y)

    if data is not None:
        bins = np.diff(data.axes.edges[0])
        data_tmp = copy.deepcopy(data)
        for i in range(data.size - 2):
            data_tmp[i] *= 10 / bins[i]
        data_tmp.plot(
            ax=axes,
            yerr=False,
            flow=None,
            histtype="fill",
            alpha=0.3,
            label="Data",
            color=vset.blue,
        )

    for pdf, c, l in zip(pdfs, colors, labels):
        bins = np.diff(pdf.axes.edges[0])
        pdf_tmp = copy.deepcopy(pdf)

        for i in range(pdf.size - 2):
            pdf_tmp[i] *= 10 / bins[i]

        pdf_tmp.plot(
            ax=axes, linewidth=0.6, color=c, yerr=False, flow=None, histtype="step", label=l
        )

    axes.legend(loc="best", edgecolor="black", frameon=True, facecolor="white", framealpha=1)

    save_pdf.savefig()
    # plt.show()


def plot_mc(
    pdf,
    name,
    save_pdf,
    data=None,
    range_x=(500, 4000),
    range_y=(0.01, 5000),
    pdf2=None,
    scale="log",
    label="",
):
    """Plot the MC files"""
    vset = tc.tol_cset("vibrant")

    fig, axes = lps.subplots(1, 1, figsize=(5, 4), sharex=True, gridspec_kw={"hspace": 0})
    axes.set_xlabel("Energy [keV]")
    axes.set_ylabel("counts/10 keV")
    axes.set_title(name)
    axes.set_yscale(scale)
    axes.set_xlim(range_x)
    axes.set_ylim(range_y)
    if data is not None:
        data.plot(
            ax=axes,
            yerr=False,
            flow=None,
            histtype="fill",
            alpha=0.3,
            label="Data",
            color=vset.blue,
        )
    pdf.plot(
        ax=axes, linewidth=1, color=vset.red, yerr=False, flow=None, histtype="step", label=label
    )
    if pdf2 is not None:
        pdf2.plot(
            ax=axes,
            linewidth=1,
            color=vset.red,
            alpha=0.6,
            yerr=False,
            flow=None,
            histtype="step",
            label="90 pct upper limit",
        )

    if label != "":
        axes.legend(loc="best", edgecolor="black", frameon=True, facecolor="white", framealpha=1)

    save_pdf.savefig()

    # plt.close()


def vals2hist(vals, hist):
    for i in range(hist.size - 2):
        hist[i] = vals[i]
    return hist


def slow_convolve(priors, pdfs, rvs, n_samp=10000):
    """
    Convolution done in a slow way but fast enough
    """
    ### get the number of bins and components

    n_bins = 0
    n_comp = 0
    for comp in priors["components"]:
        vals = np.array(pdfs[comp["name"]].values())
        n_bins = len(vals)
        n_comp += 1

    output = np.zeros((n_bins, n_samp))

    for i in range(n_samp):
        h_tot = np.zeros(n_bins)
        for comp in priors["components"]:
            rv = rvs[comp["name"]]

            a = rv.rvs(1)

            a = np.array(a)
            h_tot += np.array(pdfs[comp["name"]].values()) * a[0]

        output[:, i] = h_tot

    return output


def plot_pdf(rv, high, samples=np.array([]), pdf_obj=None, name=""):
    """Plot the PDF and optionally some samples"""

    vset = tc.tol_cset("vibrant")

    fig, axes = lps.subplots(1, 1, figsize=(5, 4), sharex=True, gridspec_kw={"hspace": 0})

    x = np.linspace(0, high, 10000)

    pdf = rv.pdf(x)
    axes.plot(x, pdf, color=vset.red, label="prior distribution")

    if len(samples) > 0:
        axes.hist(
            samples,
            range=(0, high),
            bins=100,
            label="Samples",
            density=True,
            color=vset.blue,
            alpha=0.4,
        )
    axes.set_xlabel("decays/yr")
    axes.legend(loc="best")
    axes.set_ylabel("Probability Density ")
    axes.set_xlim(0, high)
    axes.set_title(name)
    if pdf_obj is not None:
        pdf_obj.savefig()

    plt.close()


def scale_hist(hist, scale):
    """Scale a hist object"""
    hist_scale = copy.deepcopy(hist)
    for i in range(hist.size - 2):
        hist_scale[i] *= scale

    return hist_scale


def get_hist(obj, range: tuple = (132, 4195), bins: int = 10):
    """
    Extract the histogram (hist package object) from the uproot histogram
    Parameters:
        - obj: the uproot histogram
        - range: (tuple): the range of bins to select (in keV)
        - bins (int): the (constant) rebinning to apply
    Returns:
        - hist
    """
    return obj.to_hist()[range[0] : range[1]][hist.rebin(bins)]


def get_hist_variable(obj, range: tuple = (132, 4195), bins: list = []):
    """
    Extract the histogram (hist package object) from the uproot histogram and variably rebin
    Parameters:
        - obj: the uproot histogram
        - range: (tuple): the range of bins to select (in keV)
        - bins (list): list of bin edges
    Returns:
        - hist with a variable binning
    """

    return variable_rebin(obj.to_hist()[range[0] : range[1]], bins)


def get_data(path, spectrum="mul_surv", det_type="all", r=(0, 4000), b=10):
    """Get the histogram (PDF) and the number of simulated primaries (with uproot)"""

    file = uproot.open(path, object_cache=None)

    if f"{spectrum}/{det_type}" in file:
        hist = file[f"{spectrum}/{det_type}"]
        hist = get_hist(hist, range=r, bins=b)

    else:
        raise ValueError(f"Error: {spectrum}/{det_type} not in {path}")
    return hist


def get_pdf_and_norm(path, spectrum="mul_surv", det_type="all", r=(0, 4000), b=10):
    """Get the histogram (PDF) and the number of simulated primaries (with uproot)"""

    file = uproot.open(path, object_cache=None)

    if f"{spectrum}/{det_type}" in file:
        hist = file[f"{spectrum}/{det_type}"]
        hist = get_hist(hist, range=r, bins=b)
        N = int(file["number_of_primaries"])

    else:
        raise ValueError(f"Error: {spectrum}/{det_type} not in {path}")
    return hist, N


def get_counts_minuit(counts: np.ndarray, energies: np.ndarray):
    """
    A basic counting analysis implemented in Minuit, this implements a 3 bin counting analysis
    Parameters:
        - counts (np.ndarray): a numpy array of the counts -must have 3 elements
        - energies (np.ndarray): an array of the bin edges - must have 4 elements
    Returns:
        -(best fit number of counts, low error, high error)
    """
    if len(counts) != 3:
        raise ValueError("Counts array must have 3 element")
    if len(energies) != 4:
        raise ValueError("Energies array must have 3 element")

    cost = create_counting_likelihood(counts, np.diff(energies))
    bins = np.diff(energies)
    Ns_guess = abs(counts[1] - bins[1] * counts[0] / bins[0]) + 10

    Nb_guess = counts[0]
    guess = (Ns_guess, Nb_guess)
    m = Minuit(cost, *guess)
    m.limits[1] = (0, 1.5 * Nb_guess + 10)
    m.limits[0] = (0, 2 * Ns_guess + 10)
    m.migrad()

    if m.valid:
        m.minos()
        elow = abs(m.merrors["Ns"].lower)
        ehigh = abs(m.merrors["Ns"].upper)
    else:
        elow = m.errors["Ns"]
        ehigh = m.errors["Ns"]

    return m.values["Ns"], elow, ehigh


def get_peak_counts(
    peak: float,
    name_peak: str,
    data_file: str,
    livetime: float = 1,
    spec: str = "mul_surv",
    size: float = 5,
):
    """Get the counts in a peak in the data, it uses a basic counting analysis with minuit.
    Parameters:
        - peak: (float) the energy of the peak
        - name_peak (str) the name of the peak
        - data_file (str): the file of the data
        - livetime (float) detector livetime
        - spec (str): spectrum to look for the peak in (default mul_surv)
        - size (float): the size of the center region (half)
    Returns
        - (count rate, low error and high error) - all divided by livetime
    """

    regions = {name_peak: [[peak - size, peak + size]]}
    left_sideband = {name_peak: [[peak - 3 * size, peak - size]]}
    right_sideband = {name_peak: [[peak + size, peak + 3 * size]]}

    det_types, name, Ns = get_det_types("all")

    energies = np.array(
        [
            left_sideband[name_peak][0][0],
            left_sideband[name_peak][0][1],
            regions[name_peak][0][1],
            right_sideband[name_peak][0][1],
        ]
    )

    data_counts = get_data_counts_total(spec, det_types, regions, data_file, key_list=[name_peak])
    data_counts_right = get_data_counts_total(
        spec, det_types, right_sideband, data_file, key_list=[name_peak]
    )
    data_counts_left = get_data_counts_total(
        spec, det_types, left_sideband, data_file, key_list=[name_peak]
    )

    ### the efficiency calculation
    counts, low, high = get_counts_minuit(
        np.array(
            [
                data_counts_left["all"][name_peak],
                data_counts["all"][name_peak],
                data_counts_right["all"][name_peak],
            ]
        ),
        energies,
    )

    return counts / livetime, low / livetime, high / livetime


def get_eff_minuit(counts_total, counts_after, energies):
    """
    Get the efficiency with Minuit the likelihood is just a binned one



    """

    cost = create_efficiency_likelihood(counts_after, counts_total - counts_after)
    Ns_guess = counts_total[1]
    Nb_guess = counts_total[0]

    eff_s = counts_after[1] / counts_total[1]

    eff_b = counts_after[0] / counts_total[0]
    guess = (Ns_guess, Nb_guess, eff_s, eff_b)
    m = Minuit(cost, *guess)
    m.limits[1] = (0, 1.5 * Nb_guess + 10)
    m.limits[0] = (0, 2 * Ns_guess + 10)
    m.limits[2] = (0, 1)
    m.limits[3] = (0, 1)

    m.migrad()
    m.minos()
    elow = abs(m.merrors["eff_s"].lower)
    ehigh = abs(m.merrors["eff_s"].upper)
    plot_eff_calc(energies, counts_after, counts_total - counts_after, m.values)
    return m.values["eff_s"], elow, ehigh


def get_data_counts(spectrum: str, det_type: str, regions: dict, file):
    """

    Function to get the counts in the data in a range:
    Parameters:
        - spectrum (str): the spectrum to use
        - det_type (str): the detector type to select
        - regions (dict): dictonary of regions to get counts in
        - file: uproot file of the data
    Returns:
        - dict of the data counts

    """

    if f"{spectrum}/{det_type}" in file:
        hist = file[f"{spectrum}/{det_type}"]
    else:
        data_counts = {}
        warnings.warn(f"{spectrum}/{det_type} not in data")
        for region in regions:
            data_counts[region] = 0
        return data_counts

    data_counts = {}
    hist = hist.to_hist()
    for region, rangel in regions.items():
        data = 0
        for i in range(len(rangel)):
            data += float(integrate_hist(hist, rangel[i][0], rangel[i][1]))

        data_counts[region] = data
    print(data_counts)
    return data_counts


def name2number(meta: LegendMetadata, name: str):
    """
    Function to get the channel number given the name.
    Parameters:
        meta (LegendMetadata): An object containing metadata information.
        name (str): The channel name to be converted to a channel number
    Returns:
        str: The channel numeber corresponding to the provided channel name

    Raises:
        ValueError: If the provided channel name does not correspond to any detector number


    """

    meta = LegendMetadata("../legend-metadata")

    chmap = meta.channelmap(datetime.now())

    if name in chmap:
        return f"ch{chmap[name].daq.rawid:07}"
    raise ValueError(f"Error detector {name} does not have a number ")


def number2name(meta: LegendMetadata, number: str) -> str:
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
        if f"ch{chmap[detector].daq.rawid:07}" == number:
            return detector
    raise ValueError(f"Error detector {number} does not have a name")


def get_channel_floor(name: str, groups_path: str = "cfg/level_groups_Sofia.json") -> str:
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

    with open(groups_path) as json_file:
        level_groups = json.load(json_file)

    for key, chan_list in level_groups.items():
        if name in chan_list:
            return key

    raise ValueError(f"Channel {name} has no position ")


def get_channels_map():
    """Get the channels map
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

    string_channels = {}
    string_types = {}
    for string in range(1, 12):
        if string == 6:
            continue
        channels_names = [
            f"ch{chmap[detector].daq.rawid:07}"
            for detector, dic in meta.dataprod.config.on(time)["analysis"].items()
            if dic["processable"] == True
            and chmap[detector]["system"] == "geds"
            and chmap[detector]["location"]["string"] == string
        ]
        channels_types = [
            chmap[detector]["type"]
            for detector, dic in meta.dataprod.config.on(time, system="cal")["analysis"].items()
            if dic["processable"] == True
            and chmap[detector]["system"] == "geds"
            and chmap[detector]["location"]["string"] == string
        ]

        string_channels[string] = channels_names
        string_types[string] = channels_types
    print(string_channels)
    return string_channels, string_types


def get_exposure(group: list, periods: list = ["p03", "p04", "p06", "p07"]) -> float:
    """
    Get the livetime for a given group a list of strs of channel names or detector types or "all"
    Parameters:
        group: list of str
        periods: list of periods to consider default is 3,4,6,7 (vancouver dataset)
    Returns:
        - a float of the exposure (summed over the group)
    """

    meta = LegendMetadata("../legend-metadata")

    with open("cfg/analysis_runs.json") as file:
        analysis_runs = json.load(file)
    ### also need the channel map

    groups_exposure = np.zeros(len(group))

    for period in periods:
        if period not in analysis_runs.keys():
            continue
        ## loop over runs
        for run, run_dict in meta.dataprod.runinfo[period].items():
            ## skip 'bad' runs
            if run in analysis_runs[period]:
                ch = meta.channelmap(run_dict["phy"]["start_key"])

                for i in range(len(group)):
                    item = group[i]

                    if item == "all":
                        geds_list = [
                            _name
                            for _name, _dict in ch.items()
                            if ch[_name]["system"] == "geds"
                            and ch[_name]["analysis"]["usability"] in ["on", "no_psd"]
                        ]
                    elif (
                        (item == "bege") or (item == "icpc") or (item == "coax") or (item == "ppc")
                    ):
                        geds_list = [
                            _name
                            for _name, _dict in ch.items()
                            if ch[_name]["system"] == "geds"
                            and ch[_name]["type"] == item
                            and ch[_name]["analysis"]["usability"] in ["on", "no_psd"]
                        ]
                    else:
                        geds_list = [
                            _name
                            for _name, _dict in ch.items()
                            if ch[_name]["system"] == "geds"
                            and f"ch{ch[_name]['daq']['rawid']}" == item
                            and ch[_name]["analysis"]["usability"] in ["on", "no_psd"]
                        ]

                    ## loop over detectors
                    for det in geds_list:
                        mass = ch[det].production.mass_in_g / 1000
                        if "phy" in run_dict:
                            groups_exposure[i] += mass * (
                                run_dict["phy"]["livetime_in_s"] / (60 * 60 * 24 * 365.25)
                            )

    return np.sum(groups_exposure)


def get_livetime(verbose: bool = True) -> float:
    """Get the livetime from the metadata
    Parameters:
        verbose: (bool), default True a bool to say if the livetime is printed to the screen
    Returns:
        - float : the livetime in s

    """

    meta = LegendMetadata()

    with open("cfg/analysis_runs.json") as file:
        analysis_runs = json.load(file)

    analysis_runs = meta.dataprod.config.analysis_runs

    taup = 0
    vancouver = 0
    neutrino = 0
    p8 = 0
    p9 = 0
    ### loop over periods
    for period in meta.dataprod.runinfo.keys():
        livetime_tot = 0

        ## loop over runs
        for run in meta.dataprod.runinfo[period].keys():
            ## skip 'bad' runs
            if period in analysis_runs.keys() and run in analysis_runs[period]:
                if (
                    "phy" in meta.dataprod.runinfo[period][run].keys()
                    and "livetime_in_s" in meta.dataprod.runinfo[period][run]["phy"].keys()
                ):
                    print(meta.dataprod.runinfo[period][run]["phy"])
                    time = meta.dataprod.runinfo[period][run]["phy"]["livetime_in_s"]
                    livetime_tot += time

        ### sum the livetimes
        if period == "p03" or period == "p04":
            taup += livetime_tot
            vancouver += livetime_tot
        if period == "p06" or period == "p07":
            vancouver += livetime_tot
        if period == "p08":
            p8 += livetime_tot
        if period == "p09":
            p9 += livetime_tot

        neutrino += livetime_tot

    if verbose:
        print(f"Taup livetime = {taup/(60*60*24*365.25)}")
        print(f"Vancouver livetime = {vancouver/(60*60*24*365.25)}")
        print(f"Vancouver Full livetime = {(vancouver+p8+p9)/(60*60*24*365.25)}")
        print(f"nu24 livetime = {neutrino/(60*60*24*365.25)}")
    return vancouver


def get_det_types(
    group_type: str,
    string_sel: int = None,
    det_type_sel: str = None,
    level_groups: str = "cfg/level_groups_Sofia.json",
) -> (dict, str, list):
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
    Raises:
    """

    ### extract the meta data
    meta = LegendMetadata("../legend-metadata")
    Ns = []

    ### just get the total summed over all channels
    if group_type == "sum" or group_type == "all":
        det_types = {
            "all": {
                "names": ["icpc", "bege", "ppc", "coax"],
                "types": ["icpc", "bege", "ppc", "coax"],
            }
        }
        namet = "all"
    elif group_type == "types":
        det_types = {
            "icpc": {"names": ["icpc"], "types": ["icpc"]},
            "bege": {"names": ["bege"], "types": ["bege"]},
            "ppc": {"names": ["ppc"], "types": ["ppc"]},
            "coax": {"names": ["coax"], "types": ["coax"]},
        }
        namet = "by_type"
    elif group_type == "string":
        string_channels, string_types = get_channels_map()
        det_types = {}

        ### loop over strings
        for string in string_channels.keys():
            det_types[string] = {"names": [], "types": []}

            for dn, dt in zip(string_channels[string], string_types[string]):
                if det_type_sel == None or det_type_sel == dt:
                    det_types[string]["names"].append(dn)
                    det_types[string]["types"].append(dt)

        namet = "string"

    elif group_type == "chan":
        string_channels, string_types = get_channels_map()
        det_types = {}
        namet = "channels"
        Ns = []
        N = 0
        for string in string_channels.keys():
            if string_sel == None or string_sel == string:
                chans = string_channels[string]
                types = string_types[string]

                for chan, type in zip(chans, types):
                    if det_type_sel == None or type == det_type_sel:
                        det_types[chan] = {"names": [chan], "types": [type]}
                        N += 1
            Ns.append(N)

    elif group_type == "floor":
        string_channels, string_types = get_channels_map()
        groups = ["top", "mid_top", "mid", "mid_bottom", "bottom"]
        det_types = {}
        namet = "floor"
        for group in groups:
            det_types[group] = {"names": [], "types": []}

        for string in string_channels.keys():
            channels = string_channels[string]

            ## loop over channels per string
            for i in range(len(channels)):
                chan = channels[i]
                name = number2name(meta, chan)
                group = get_channel_floor(name, groups_path=level_groups)
                if string_sel == None or string_sel == string:
                    if det_type_sel == None or string_types[string][i] == det_type_sel:
                        det_types[group]["names"].append(chan)
                        det_types[group]["types"].append(string_types[string][i])
    else:
        raise ValueError("group type must be either floor, chan, string sum all or types")

    ### get the exposure
    ### --------------
    for group in det_types:
        list_of_names = det_types[group]["names"]
        exposure = get_exposure(list_of_names)

        det_types[group]["exposure"] = exposure

    return det_types, namet, Ns


def select_region(histo, cuts: list = []):
    """
    Function to select a region of the histo
    Parameters:
        - histo: hist histogram object
        - cuts (list) a list of cuts each of which are dictonaries
    Returns:
        - histogram with the cut applied

    """
    histo_out = copy.deepcopy(histo)
    energy_2 = histo_out.axes.centers[0]
    energy_1 = histo_out.axes.centers[1]
    matrix_e2, matrix_e1 = np.meshgrid(energy_2, energy_1, indexing="ij")
    energy_1 = energy_1[0]
    energy_2 = energy_2.T[0]

    matrix_sum = energy_2[:, np.newaxis] + energy_1
    matrix_diff = -energy_2[:, np.newaxis] + energy_2
    logic = np.ones_like(matrix_sum, dtype="bool")

    for cut in cuts:
        if cut["var"] == "e1":
            matrix = matrix_e1
        elif cut["var"] == "e2":
            matrix = matrix_e2
        elif cut["var"] == "diff":
            matrix = matrix_diff
        else:
            matrix = matrix_sum

        if cut["greater"] == True:
            logic_tmp = matrix > cut["val"]
        else:
            logic_tmp = matrix <= cut["val"]

        logic = (logic) & (logic_tmp)

    w, x, y = histo_out.to_numpy()
    w_new = w
    w_new[~logic] = 0

    for i in range(len(histo_out.axes.centers[0])):
        for j in range(len(histo_out.axes.centers[1])):
            histo_out[i, j] = w_new[i, j]
    del histo

    return histo_out


def project_sum(histo):
    """Take a 2D histogram and create a 1D histogram with the sum of the bin contents
    Note: This is in principle impossible here an approximation is used where a sum of a -- a+1 and b-- b+1 is split evenly between bin a+b -- a+b+1 and a+b+1 --- a+b +2
    If you want a proper summed histo you should build it directly from the raw data
    Parameters:
        -histo: 2D histogram to be projected
    Returns:
        - projected 1D histogram
    """
    w, x, y = histo.to_numpy()

    hist_energy_sum = Hist.new.Reg(int((len(x) - 2) * 2) + 2, 0, x[-1] + y[-1]).Double()

    rows, cols = w.shape

    indices = np.arange(rows) + np.arange(cols)[:, None]

    diagonal_sums = np.bincount(indices.flatten(), weights=w.flatten())

    for i in range(hist_energy_sum.size - 3):
        hist_energy_sum[i] += diagonal_sums[i]

    return hist_energy_sum


def fast_variable_rebinning(x, y, weights, edges_x, edges_y):
    """
    Perform a variable rebinning of a 2D histogram in a fast way with numpy operations:
    Parameters:
        - x :np.array of bin centers in x
        - y :np.array of bin centers in y
        - weights: 2D np.array of the weights for the histo
        - edges_x,edges_y np.arrays of the bin edges to rebin to
    Returns
        - hist object of the rebinnined histogram
    """
    indices_x = np.searchsorted(edges_x, x[0:-1] + 0.1) - 1
    indices_y = np.searchsorted(edges_y, y[0:-1] + 0.1) - 1

    flat_indices = indices_x[:, np.newaxis] * len(edges_y) + indices_y
    flat_indices = flat_indices.flatten()

    num_bins_x = len(edges_x)
    num_bins_y = len(edges_y)

    hist = np.zeros((num_bins_x, num_bins_y)).flatten()

    np.add.at(hist, flat_indices, weights.flatten())

    wrb = hist.reshape((num_bins_x, num_bins_y))

    histo_var = Hist.new.Variable(edges_x).Variable(edges_y).Double()

    for i in range(int(len(np.hstack(histo_var.axes.centers[0])))):
        for j in range(int(len(histo_var.axes.centers[1][0]))):
            histo_var[i, j] = wrb[i, j]

    return histo_var


def variable_rebin(histo, edges: list):
    """
    Perform a variable rebinning of a hist object
    Parameters:
        - histo: The histogram object
        - edges: The list of the bin edges
    Returns:
        - variable rebin histo
    """
    histo_var = Hist.new.Variable(edges).Double()
    for i in range(histo.size - 2):
        cent = histo.axes.centers[0][i]

        histo_var[cent * 1j] += histo.values()[i]

    return histo_var


def variable_rebin_2D(histo, edges_x, edges_y):
    """
    Perform a variable rebinning of a 2D histogram
    Done in a slow way with a loop.
    Parameters
        -histo (Hist object)
        -edges_x,edges_y: np.array of the bin_edges
    Returns:
        - rebinnined histogram
    """

    histo_var = Hist.new.Variable(edges_x).Variable(edges_y).Double()

    xarr = np.hstack(histo.axes.centers[0])
    vals = histo.values()
    for i in range(int(len(np.hstack(histo.axes.centers[0])))):
        if i % 100 == 0:
            print(f"Progreess = {100*i/len(np.hstack(histo.axes.centers[0])):02f} %")
        for j in range(int(len(histo.axes.centers[1][0]))):
            cent_x = xarr[i]
            cent_y = histo.axes.centers[1][0][j]
            histo_var[cent_x * 1j, cent_y * 1j] += vals[i, j]

    return histo_var


def normalise_histo_2D(histo, factor=1):
    """
    Normalise 2D histogram by the bin area:
    Parameters:
    ----------------------
        - histo: Hist object
        - factor: a scaling factor to multiply the histo by
    Returns
    ----------------------
        - normalised histo
    """
    widths_x = np.diff(np.hstack(histo.axes.edges[0]))
    widths_y = np.diff(histo.axes.edges[1][0])

    for i in range(int(len(np.hstack(histo.axes.centers[0])))):
        if i % 100 == 0:
            print(f"Progreess = {100*i/len(np.hstack(histo.axes.centers[0])):02f} %")
        for j in range(int(len(histo.axes.centers[1][0]))):
            if i < 4 or j < 4:
                histo[i, j] = 0
            else:
                histo[i, j] /= widths_x[i] * widths_y[j]
                histo[i, j] *= factor

    return histo


def normalise_histo(hist, factor=1):
    """
    Normalise a histogram into units of counts/keV (by bin width)
     Parameters:
    ----------------------
        - histo: Hist object
        - factor: a scaling factor to multiply the histo by
    Returns
    ----------------------
        - normalised histo
    """

    widths = np.diff(hist.axes.edges[0])

    for i in range(hist.size - 2):
        hist[i] /= widths[i]
        hist[i] *= factor

    return hist
