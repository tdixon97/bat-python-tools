"""
A script to plot the fit reconstruction of the LEGEND / hmixfit background model
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk) but based on a script from Luigi Pertoldi.
"""

from __future__ import annotations

import shutil

from legend_plot_style import LEGENDPlotStyle as lps

lps.use("legend")
import argparse
import json
import os
import subprocess
from collections import OrderedDict

import hist
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
import uproot
import utils
from hist import Hist
from matplotlib.backends.backend_pdf import PdfPages

vset = tc.tol_cset("vibrant")
mset = tc.tol_cset("muted")
plt.rc("axes", prop_cycle=plt.cycler("color", list(vset)))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.6,
}


def remove_duplicated(listo, range):
    list_new = []
    range_new = []
    for item, range_tmp in zip(listo, range):
        if item not in list_new:
            list_new.append(item)
            range_new.append(range_tmp)
    return list_new, range_new


##### the old set of arguments
##### --------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="A script with command-line argument.")
parser.add_argument(
    "-C",
    "--components",
    type=str,
    help="json components config file path",
    default="cfg/components.json",
)
parser.add_argument("-r", "--regions", type=int, help="Shade regions on the plot", default=0)
parser.add_argument("-w", "--width", type=int, help="width for canvas", default=6)
parser.add_argument("-a", "--save", type=int, help="save the output (1) or print (0)", default=1)
parser.add_argument("-c", "--config", type=str, help="hmixfit configuration file for the fit")
parser.add_argument(
    "-O", "--outdir", type=str, help="output directory to save plots", default="plots/summary/"
)
parser.add_argument(
    "-o",
    "--originals",
    type=bool,
    help="Boolean flag to plot the original uniform binned histograms (in 1 keV binning)",
    default=False,
)


# TODO:
# 1) make the code figure out fit name automatically
### read the arguments


args = parser.parse_args()
cfg = args.config
outdir = args.outdir

with open(cfg) as json_file:
    cfg_dict = json.load(json_file)

### extract all we need from the cfg dict
fit_name, out_dir, det_types, ranges, dataset_names, ds_files = utils.parse_cfg(cfg_dict)

det_types, ranges = remove_duplicated(det_types, ranges)


outfile = out_dir + "/hmixfit-" + fit_name + "/histograms.root"
os.makedirs(f"{outdir}/{fit_name}/", exist_ok=True)


### copy the knowledge update plots
comps = 0
for key, object in cfg_dict["fit"]["parameters"].items():
    if "fixed" not in object:
        comps += 1

### -------------------------------------------------------
shutil.copy2(out_dir + "/hmixfit-" + fit_name + "/know-update.pdf", f"{outdir}/{fit_name}/")
command = (
    f"pdftk {outdir}/{fit_name}/know-update.pdf cat 1-{comps} output {outdir}/{fit_name}/tmp.pdf"
)
subprocess.run(command, shell=True, check=False)
subprocess.run(
    f"mv  {outdir}/{fit_name}/tmp.pdf {outdir}/{fit_name}/know-update.pdf", shell=True, check=False
)

bin = 1
## could replace with nicer format
type_names = [
    d.replace("mul_surv", "M1")
    .replace("mul2_surv", "M2")
    .replace("cat", "")
    .replace("/", "")
    .replace("sum", "s")
    for d in det_types
]

json_file = args.components

exclude_range = (2014, 2064)
width = args.width
comp_name = json_file[:-5]
shade_regions = args.regions
save = bool(args.save)
ylowlim = 0.001

regions = {
    "$2\\nu\\beta\\beta$": (800, 1300),
    "K40": (1455, 1465),
    "K42": (1520, 1530),
    "Tl compton": (1900, 2500),
    "Tl peak": (2600, 2630),
}
colors = [vset.red, vset.orange, vset.magenta, vset.teal, "grey"]

get_originals = args.originals

### options to run this code

#### creat the gamma lines plots
### ---------------------------
blind = True

gamma_line_plots = []
for d in dataset_names:
    gamma_line_plots.append(
        {
            "Tl": {
                "lines": [583, 2615],
                "data": [],
                "model": [],
                "groups": [0],
                "names": [],
                "count": 0,
            },
            "Bi": {
                "lines": [609, 1764, 2204],
                "data": [],
                "model": [],
                "groups": [0],
                "names": [],
                "count": 0,
            },
            "K": {
                "lines": [1461, 1525],
                "data": [],
                "model": [],
                "groups": [0],
                "names": [],
                "count": 0,
            },
        }
    )


with open(json_file) as file:
    components = json.load(file, object_pairs_hook=OrderedDict)


def get_hist(obj):
    return obj.to_hist()[hist.rebin(bin)]


def get_hist_rb(obj):
    return obj.to_hist()


### save to one PDF

with PdfPages(f"{outdir}/{fit_name}/fit_reconstructions.pdf") as pdf:
    ## loop over detector type
    for det_type, type_name, fit_range in zip(det_types, type_names, ranges):
        xlow = fit_range[0]
        xhigh = fit_range[1]

        labels = [type_name]
        ## set height of histo
        y = 5
        if det_type != "multi":
            y = 2.5

        ### create the plot
        with uproot.open(outfile) as f:
            datasets = [
                "{}_{}".format(dataset_name, det_type.split("/")[1])
                for dataset_name in dataset_names
            ]

            ## loop over datasets
            hs = {}
            # for ds, ax, title in zip(datasets, axes.flatten(), labels):
            for i in range(len(datasets)):
                ds = datasets[i]

                if det_type == "multi":
                    title = labels[i]
                else:
                    title = labels[0]

                for comp, info in components.items():
                    if det_type != "sum" or ds == datasets[0]:
                        hs[comp] = None

                    ## loop over the contributions to h

                    for name in info["hists"]:
                        if get_originals == True:
                            if name not in f[ds]["originals"]:
                                continue
                            if hs[comp] is None:
                                w = f[ds][name].values()
                                dims = w.ndim
                                type_plots = []
                                if dims == 1:
                                    hs[comp] = {}
                                    hs[comp]["1D"] = get_hist(f[ds]["originals"][name])
                                    type_plots.append("1D")
                                else:
                                    hs[comp] = {}
                                    hs[comp]["2D_x"] = get_hist(f[ds]["originals"][name]).project(0)
                                    hs[comp]["2D_y"] = get_hist(f[ds]["originals"][name]).project(1)
                                    type_plots.append("2D_x")
                                    type_plots.append("2D_y")
                            else:
                                ### get the number of dimensions
                                w = f[ds]["originals"][name].values()
                                dims = w.ndim
                                if dims == 1:
                                    hs[comp]["1D"] += get_hist_rb(f[ds]["originals"][name])
                                else:
                                    hs[comp]["2D_x"] += get_hist_rb(
                                        f[ds]["originals"][name]
                                    ).project(0)
                                    hs[comp]["2D_y"] += get_hist_rb(
                                        f[ds]["originals"][name]
                                    ).project(1)
                        else:
                            if name not in f[ds]:
                                # raise ValueError("PDF {} not in f[{}]".format(name,ds))
                                continue

                            if hs[comp] is None:
                                ### get the number of dimensions
                                w = f[ds][name].values()
                                dims = w.ndim
                                type_plots = []
                                if dims == 1:
                                    hs[comp] = {}
                                    hs[comp]["1D"] = get_hist_rb(f[ds][name])
                                    type_plots.append("1D")
                                else:
                                    hs[comp] = {}
                                    hs[comp]["2D_x"] = get_hist_rb(f[ds][name]).project(0)
                                    hs[comp]["2D_y"] = get_hist_rb(f[ds][name]).project(1)
                                    type_plots.append("2D_x")
                                    type_plots.append("2D_y")

                            else:
                                ### get the number of dimensions
                                w = f[ds][name].values()
                                dims = w.ndim
                                if dims == 1:
                                    hs[comp]["1D"] += get_hist_rb(f[ds][name])
                                else:
                                    hs[comp]["2D_x"] += get_hist_rb(f[ds][name]).project(0)
                                    hs[comp]["2D_y"] += get_hist_rb(f[ds][name]).project(1)

                ### now make the plot
                ### ---------------------------------
                for type_plot in type_plots:
                    fig, axes_full = lps.subplots(
                        2,
                        1,
                        figsize=(width, y),
                        sharex=True,
                        gridspec_kw={"height_ratios": [8, 2], "hspace": 0},
                    )

                    for comp, info in components.items():
                        if hs[comp] is None:
                            continue

                        ### scale for bin width
                        bin_widths = np.diff(hs[comp][type_plot].axes.edges[0])

                        if det_type == "sum" and ds != datasets[-1]:
                            continue

                        ### make the residual plot

                        if comp == "data":
                            data = hs[comp][type_plot].values()
                            bins = hs[comp][type_plot].axes.centers[0]
                            bin_widths = np.diff(hs[comp][type_plot].axes.edges[0])
                        if comp == "total_model":
                            pred = hs[comp][type_plot].values()
                            bin_widths = np.diff(hs[comp][type_plot].axes.edges[0])

                        ### rescale bin contents
                        for b in range(hs[comp][type_plot].size - 2):
                            E = hs[comp][type_plot].axes.centers[0][b]

                            if hs[comp][type_plot][b] == 0:
                                hs[comp][type_plot][b] = 1.05 * ylowlim

                            ### scale the value to be in units of cts/10 keV
                            if get_originals == False:
                                hs[comp][type_plot][b] *= 10.0 / bin_widths[b]

                        hs[comp][type_plot].plot(ax=axes_full[0], **style, **info["style"])

                    ### ----------- save gamma lines info --------
                    ### if the peak belongs to a gamma line save it
                    low = 4000
                    high = 0
                    for b in range(hs["total_model"][type_plot].size - 2):
                        E = hs["total_model"][type_plot].axes.centers[0][b]
                        bw = bin_widths[b]
                        ### get first and last non-0 bin

                        if (hs["total_model"][type_plot][b].value * bw / 10 > 1.1 * ylowlim) and (
                            low > E
                        ):
                            low = E
                        if (hs["total_model"][type_plot][b].value * bw / 10 > 1.1 * ylowlim) and (
                            high < E
                        ):
                            high = E
                        if hs["total_model"][type_plot][b].value * bw / 10 < 1.1:
                            continue
                        bw = bin_widths[b]
                        ### loop over different gamma plots

                        for plot_type, gamma_info in gamma_line_plots[i].items():
                            for gamma_counter in range(len(gamma_info["lines"])):
                                if (gamma_counter) < len(gamma_info["lines"]) and (
                                    abs(E - gamma_info["lines"][gamma_counter]) < bin_widths[b] / 2
                                ):
                                    gamma_info["model"].append(
                                        hs["total_model"][type_plot][b].value * bw / 10
                                    )
                                    if isinstance(hs["data"][type_plot][b], float):
                                        gamma_info["data"].append(
                                            hs["data"][type_plot][b] * (bw) / 10
                                        )
                                    else:
                                        gamma_info["data"].append(
                                            hs["data"][type_plot][b].value * (bw) / 10
                                        )

                                    gamma_info["names"].append(
                                        "{}".format(str(gamma_info["lines"][gamma_counter]))
                                    )
                                    gamma_info["count"] += 1

                    for plot_type, gamma_info in gamma_line_plots[i].items():
                        gamma_info["groups"].append(gamma_info["count"])

                    if det_type == "sum" and ds != datasets[-1]:
                        continue

                    ### compute residuals
                    if get_originals == True:
                        residual = np.array(
                            [((d - m) / (d**0.5)) if d > 0.5 else 0 for d, m in zip(data, pred)]
                        )
                    else:
                        residual = []
                        for d, m, s in zip(data, pred, bin_widths):
                            obs = d * s / 10
                            mu = m * s / 10
                            resid = utils.normalized_poisson_residual(mu, obs)
                            residual.append(resid)
                        residual = np.array(residual)

                    masked_values = np.ma.masked_where((bins > xhigh) | (bins < xlow), pred)
                    masked_values_data = np.ma.masked_where((bins > xhigh) | (bins < xlow), data)

                    maxi = max(masked_values.max(), masked_values_data.max())
                    max_y = maxi + 2 * np.sqrt(maxi) + 1

                    idx = 0

                    ### add a shaded region
                    if shade_regions == True:
                        for region in regions:
                            range = regions[region]
                            axes_full[0].fill_between(
                                np.array([range[0], range[1]]),
                                np.array([max_y, max_y]),
                                label=region,
                                alpha=0.3,
                                color=colors[idx],
                                linewidth=0,
                            )
                            idx += 1

                    ### show some excluded region (eg ROI)
                    ### annotate plot

                    legend = axes_full[0].legend(
                        loc="upper right",
                        edgecolor="black",
                        frameon=True,
                        facecolor="white",
                        framealpha=1,
                        ncol=1,
                        fontsize=6,
                    )
                    axes_full[0].set_legend_annotation()

                    ### annotate the type of fit
                    if det_type != "multi":
                        axes_full[0].set_legend_logo(
                            position="upper left", logo_type="preliminary", scaling_factor=10
                        )

                        if (
                            "livetime"
                            in cfg_dict["fit"]["theoretical-expectations"][ds_files[i]][det_type]
                        ):
                            livetime = cfg_dict["fit"]["theoretical-expectations"][ds_files[i]][
                                det_type
                            ]["livetime"]
                        else:
                            livetime = cfg_dict["livetime"]

                        axes_full[0].annotate(
                            f"Fit of {utils.format(title)} - livetime {livetime:0.2g} yr",
                            (0.5, 0.8),
                            xycoords="axes fraction",
                            fontsize=9,
                        )

                    ## set labels

                    if det_type == "multi":
                        axes_full[0].set_xlabel("Energy (keV)")
                    axes_full[0].set_ylabel("Counts / 10 keV")
                    axes_full[0].set_xlim(xlow, xhigh)

                    ### now plot the residual
                    if det_type != "multi":
                        axes_full[1].axhspan(-3, 3, color="red", alpha=0.5, linewidth=0)
                        axes_full[1].axhspan(-2, 2, color="gold", alpha=0.5, linewidth=0)
                        axes_full[1].axhspan(-1, 1, color="green", alpha=0.5, linewidth=0)

                        axes_full[1].errorbar(
                            bins, residual, fmt="o", color="black", markersize=0.8, linewidth=0.6
                        )
                        axes_full[1].set_xlabel("Energy (keV)")
                        axes_full[1].set_ylabel("Residual")
                        axes_full[1].set_yscale("linear")
                        axes_full[1].set_xlim(xlow, xhigh)

                        axes_full[1].set_ylim(-6, 6)
                        if exclude_range != 0:
                            axes_full[1].fill_between(
                                np.array([exclude_range[0], exclude_range[1]]),
                                y1=np.array([-6, -6]),
                                y2=np.array([6, 6]),
                                alpha=1,
                                color="grey",
                                linewidth=0,
                            )
                        axes_full[1].xaxis.set_tick_params(top=False)

                        plt.tight_layout()

                    plt.tight_layout()
                    for scale in ["linear", "log"]:
                        if scale == "log":
                            max_y_p = 50 * max_y
                        else:
                            max_y_p = 1.3 * max_y

                        axes_full[0].set_yscale(scale)
                        axes_full[0].set_ylim(bottom=ylowlim, top=max_y_p)
                        if exclude_range != 0:
                            axes_full[0].fill_between(
                                np.array([exclude_range[0], exclude_range[1]]),
                                np.array([max_y_p, max_y_p]),
                                label="Not in fit",
                                alpha=1,
                                color="grey",
                                linewidth=0,
                            )

                        if save:
                            pdf.savefig()
                        else:
                            plt.show()

    #### make the gamma line plot
    ### ---------------------------------------------------------
    for i, dn in enumerate(dataset_names):
        for gamma, gamma_data in gamma_line_plots[i].items():
            gamma_line_data = gamma_data["data"]
            gamma_line_model = gamma_data["model"]
            gamma_index_groups = gamma_data["groups"]
            hist_names = gamma_data["names"]
            fig, axes_full = lps.subplots(
                2,
                1,
                figsize=(7, y),
                sharex=True,
                gridspec_kw={"height_ratios": [8, 2], "hspace": 0},
            )
            axes = axes_full[0]
            gamma_hist = Hist.new.Reg(len(gamma_line_data), 0, len(gamma_line_data)).Double()
            gamma_hist_data = Hist.new.Reg(len(gamma_line_model), 0, len(gamma_line_model)).Double()

            for i in range(gamma_hist_data.size - 2):
                gamma_hist[i] = gamma_line_model[i]
                gamma_hist_data[i] = gamma_line_data[i]

            ### now make the histos
            gamma_hist_data.plot(ax=axes, **style, color=vset.blue, alpha=0.25, histtype="fill")

            gamma_hist.plot(ax=axes, **style, color="black")
            axes.set_xlim(0, len(gamma_line_data))
            axes.set_ylabel("counts/10 keV")
            axes.set_ylim(0.1, 1.2 * np.max(gamma_hist.values()))
            axes.set_yscale("linear")
            for x in gamma_index_groups:
                axes.axvline(x=x, linewidth=0.4)

            axes.set_xlabel("")
            axes_full[1].set_xticks(
                np.arange(len(gamma_line_model)) + 0.5, hist_names, rotation=90, fontsize=10
            )
            axes_full[1].set_ylabel("Residual")
            axes_full[1].axhspan(-3, 3, color="red", alpha=0.5, linewidth=0)
            axes_full[1].axhspan(-2, 2, color="gold", alpha=0.5, linewidth=0)
            axes_full[1].axhspan(-1, 1, color="green", alpha=0.5, linewidth=0)

            plt.tight_layout()
            c = 0
            for de in type_names:
                if gamma_index_groups[c] != gamma_index_groups[c + 1]:
                    axes.annotate(
                        de,
                        (
                            (1 / len(gamma_line_model))
                            * (gamma_index_groups[c] + gamma_index_groups[c + 1])
                            / 2,
                            0.91,
                        ),
                        xycoords="axes fraction",
                        fontsize=6,
                        ha="center",
                    )
                c += 1
            ### make a residuals

            data = gamma_hist_data.values()
            pred = gamma_hist.values()
            rs = []
            for d, p in zip(data, pred):
                rs.append(utils.normalized_poisson_residual(p, d))

            rs = np.array(rs)
            bins = gamma_hist.axes.centers[0]
            axes_full[1].errorbar(bins, rs, fmt="o", color="black", markersize=1, linewidth=1)
            axes_full[1].set_ylim(-max(4, max(abs(rs))) - 1, max(4, max(abs(rs))) + 1)
            fig.suptitle(dn)
            plt.tight_layout()
            if save:
                pdf.savefig()
            else:
                plt.show()
