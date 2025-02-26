"""
analyse-priors.py
Author: Toby Dixon
Date : 30th November 2023

This is a script to analyse the screening measurements for LEGEND-200, estimate the total bkg (from screening) etc.

"""

from __future__ import annotations

from legend_plot_style import LEGENDPlotStyle as lps

lps.use("legend")
import copy
import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
import uproot
import utils
import variable_binning
from matplotlib.backends.backend_pdf import PdfPages

vset = tc.tol_cset("vibrant")
mset = tc.tol_cset("muted")
sunset = tc.tol_cmap("sunset")
sunset = [
    "#332288",
    "#364B9A",
    "#4A7BB7",
    "#6EA6CD",
    "#98CAE1",
    "#C2E4EF",
    "#EAECCC",
    "#FDB366",
    "#F67E4B",
    "#DD3D2D",
    vset.red,
]

my_colors = [
    mset.indigo,
    vset.blue,
    vset.cyan,
    vset.teal,
    vset.red,
    vset.orange,
    vset.magenta,
    mset.rose,
    mset.wine,
    mset.purple,
]
plt.rc("axes", prop_cycle=plt.cycler("color", my_colors))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}


### first read the priors json

priors_file = "cfg/priors.json"
pdf_path = "../hmixfit/inputs/pdfs/l200a-pdfs_vancouver23_v5.0/l200a-pdfs/"
data_path = "../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v0.3.root"
ref_fit = "../hmixfit/results/hmixfit-l200a_vancouver_simple_all/histograms.root "
ref_ds = "l200a_vancouver23_dataset_v0_1_all"
spec = "mul_surv"
json_file = "cfg/components.json"
with open(json_file) as file:
    components = json.load(file, object_pairs_hook=OrderedDict)

with open(priors_file) as file_cfg:
    priors = json.load(file_cfg)

livetime = 0.36455
order = [
    "fiber_shroud",
    "sipm",
    "birds_nest",
    "wls_reflector",
    "pen_plates",
    "front_end_electronics",
    "hpge_insulators",
    "hpge_support_copper",
    "cables",
    "mini_shroud",
]
bins = 10

### get a variable binning
binning = variable_binning.compute_binning(
    gamma_energies=np.array([583, 609, 911, 1461, 1525, 2204, 2615]),
    low_energy=500,
    high_energy=4000,
    gamma_binning_high=10,
    gamma_binning_low=10,
    cont_binning=10,
)

# extract a prior plot it and generate random numbers
rvs = {}
quantiles = {}
pdfs = {}
pdfs_1 = {}
mc_list = []


###  PRIOR distributions
### -----------------------------------------------------------

with PdfPages("plots/priors/prior_distributions.pdf") as pdf:
    for comp in priors["components"]:
        rv1, high1, range_ci = utils.extract_prior(comp["prior"])

        point_est = comp["best_fit"]
        low_err = point_est - range_ci[0]
        high_err = range_ci[1] - point_est
        quantiles[comp["name"]] = (point_est, low_err, high_err)
        samples1 = np.array(rv1.rvs(size=100000))
        utils.plot_pdf(rv1, high1, samples1, pdf, name=comp["name"])
        rvs[comp["name"]] = rv1

        mc_pdf_1, N = utils.get_pdf_and_norm(pdf_path + comp["file"], b=1, r=(0, 4000))
        mc_pdf = utils.variable_rebin(mc_pdf_1, edges=binning)

        pdf_scale = utils.scale_hist(mc_pdf, 1 / N)
        pdf_scale_1 = utils.scale_hist(mc_pdf_1, 1 / N)
        pdfs[comp["name"]] = pdf_scale
        pdfs_1[comp["name"]] = pdf_scale_1
        mc_list.append({"components": {comp["name"]: {}}, "root-file": comp["file"]})


## first get a detector type map
det_types, name, Ns = utils.get_det_types("sum")


### now a regions map
peaks = ["Tl2615", "Bi1764", "Ac911", "K1461"]
energies = [2615, 1764, 911, 1461]
regions = {"fit_range": [[565, 4000]]}
left = {"fit_range": [[565, 4000]]}
right = {"fit_range": [[565, 4000]]}

for peak, energy in zip(peaks, energies):
    regions[peak] = [[energy - 5, energy + 5]]
    left[peak] = [[energy - 15, energy - 5]]
    right[peak] = [[energy + 5, energy + 15]]


### plot expected contribution per source
### ----------------------------------------------------------

with PdfPages("plots/priors/exp_contributions.pdf") as pdf:
    index_Bi = 0
    index_Tl = 0
    index_Ac = 0
    index_K = 0
    for comp in priors["components"]:
        point_est = comp["best_fit"]
        upper_limit = 0
        if point_est == 0:
            rv1, high, range_ci = utils.extract_prior(comp["prior"])
            point_est = high
            upper_limit = 1

        pdf_norm = utils.scale_hist(pdfs[comp["name"]], livetime * point_est)
        utils.plot_mc(pdf_norm, comp["name"], pdf)

        if index_Tl == 0 and upper_limit == 0 and "Tl208" in comp["name"]:
            total_Tl = copy.deepcopy(pdf_norm)
            index_Tl += 1
        elif upper_limit == 0 and "Tl208" in comp["name"]:
            total_Tl += pdf_norm
            index_Tl += 1

        if index_Bi == 0 and upper_limit == 0 and "Bi214" in comp["name"]:
            total_Bi = copy.deepcopy(pdf_norm)
            index_Bi += 1
        elif upper_limit == 0 and "Bi214" in comp["name"]:
            total_Bi += pdf_norm
            index_Bi += 1

        if index_Ac == 0 and upper_limit == 0 and "Ac228" in comp["name"]:
            total_Ac = copy.deepcopy(pdf_norm)
            index_Ac += 1
        elif upper_limit == 0 and "Ac228" in comp["name"]:
            total_Ac += pdf_norm
            index_Ac += 1

        if index_K == 0 and upper_limit == 0 and "K40" in comp["name"]:
            total_K = copy.deepcopy(pdf_norm)
            index_K += 1
        elif upper_limit == 0 and "K40" in comp["name"]:
            total_K += pdf_norm
            index_K += 1

plt.close()
total_other = None
total_K42 = None
total_K40 = None
total_alpha = None

### get the other contributions
### -----------------------------------------------------------------


with uproot.open(ref_fit) as f:
    for key in f[ref_ds]["originals"].keys():
        if (
            ("Bi" in key)
            or ("Ac" in key)
            or ("K40" in key)
            or ("fitted_data" in key)
            or ("total_model" in key)
        ):
            continue

        if total_other is None:
            total_other = utils.get_hist_variable(
                f[ref_ds]["originals"][key], bins=binning, range=(0, 4000)
            )
        else:
            total_other += utils.get_hist_variable(
                f[ref_ds]["originals"][key], bins=binning, range=(0, 4000)
            )

    hs = {}
    for comp, info in components.items():
        hs[comp] = None

        ## loop over the contributions to h

        for name in info["hists"]:
            if name not in f[ref_ds]["originals"]:
                continue
            if hs[comp] is None:
                hs[comp] = utils.get_hist_variable(
                    f[ref_ds]["originals"][name], bins=binning, range=(0, 4000)
                )
            else:
                hs[comp] += utils.get_hist_variable(
                    f[ref_ds]["originals"][name], bins=binning, range=(0, 4000)
                )


### get the total
total = copy.deepcopy(total_Tl)
total += total_Bi
total += total_Ac
total += total_K

for i in range(total.size - 2):
    total[i] += total_other[i]


output = utils.slow_convolve(priors, pdfs, rvs)
data = utils.get_data(data_path, b=1, r=(0, 4000))
data = utils.variable_rebin(data, binning)
output *= livetime
pdf_tmp = utils.vals2hist(output[:, 0], copy.deepcopy(total_Tl))
fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0})
utils.plot_mc_no_save(axes, pdf_tmp, name="", linewidth=0.6)
plt.show()

### now look at one bin
#### -----------------------------------------------

energies = pdf_tmp.axes.centers[0]
lows = []
highs = []
widths = np.diff(energies)
with PdfPages("plots/priors/bin_contents.pdf") as pdf:
    for bin in range(len(output[:, 0])):
        low = np.percentile(output[bin, :], 16)
        high = np.percentile(output[bin][:], 84)

        lows.append(low)
        highs.append(high)
        if (
            abs(energies[bin] - 2615) < 5
            or abs(energies[bin] - 583) < 5
            or abs(energies[bin] - 1764) < 5
            or abs(energies[bin] - 2204) < 5
            or abs(energies[bin] - 911) < 5
            or (abs(energies[bin] - 1461) < 5)
        ):
            fig, axes = lps.subplots(1, 1, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0})

            low_prev = np.percentile(output[bin - 1, :], 16)
            high_prev = np.percentile(output[bin - 1, :], 84)

            low_next = np.percentile(output[bin + 1, :], 16)
            high_next = np.percentile(output[bin + 1, :], 84)

            point_est_prev = 10 * (low_prev + high_prev) / (2 * widths[bin - 1])
            point_est_next = 10 * (low_next + high_next) / (2 * widths[bin + 1])
            bkg = (point_est_next + point_est_prev) / 2.0
            axes.hist(
                output[bin, :],
                histtype="step",
                color=vset.red,
                bins=100,
                range=(0, max(output[bin, :])),
                label="Distribution",
            )
            point_est = (high + low) / 2.0 - bkg
            low -= bkg
            high -= bkg

            axes.plot(
                [point_est, point_est],
                [0, axes.get_ylim()[1]],
                color=vset.red,
                label="Point estimate",
            )

            axes.axvspan(low, high, alpha=0.3, color="gray", label="68 pct. c.i. region")
            axes.set_title(f"Prediction for bin {energies[bin]}")
            axes.set_ylabel("Prob [arb]")
            axes.set_xlabel("Counts")
            axes.legend(loc="best")
            pdf.savefig()

            if abs(energies[bin] - 2615) < 5:
                pred_2615 = point_est / livetime
                error_low_2615 = (point_est - low) / livetime
                error_high_2615 = (high - point_est) / livetime
            if abs(energies[bin] - 1764) < 5:
                pred_1764 = point_est / livetime
                error_low_1764 = (point_est - low) / livetime
                error_high_1764 = (high - point_est) / livetime
            if abs(energies[bin] - 911) < 5:
                pred_911 = point_est / livetime
                error_low_911 = (point_est - low) / livetime
                error_high_911 = (high - point_est) / livetime

            if abs(energies[bin] - 1461) < 5:
                pred_1461 = point_est / livetime
                error_low_1461 = (point_est - low) / livetime
                error_high_1461 = (high - point_est) / livetime
            plt.close()


#### save a plot with the expected counts
mc_list = np.array(mc_list)
effs = utils.get_total_efficiency(det_types, None, "mul_surv", regions, pdf_path, mc_list=mc_list)
effs_left = utils.get_total_efficiency(det_types, None, "mul_surv", left, pdf_path, mc_list=mc_list)
effs_right = utils.get_total_efficiency(
    det_types, None, "mul_surv", right, pdf_path, mc_list=mc_list
)

### get counts in gamma lines (data)
### -------------------------------------------------------------------------------------------------

data_file = uproot.open(data_path)
counts_2615_data, low_2615_data, high_2615_data = utils.get_peak_counts(
    2615, "Tl208", data_file, livetime
)
counts_1764_data, low_1764_data, high_1764_data = utils.get_peak_counts(
    1764, "Bi214", data_file, livetime
)
counts_911_data, low_911_data, high_911_data = utils.get_peak_counts(
    911, "Ac228", data_file, livetime
)
counts_1461_data, low_1461_data, high_1461_data = utils.get_peak_counts(
    1461, "K40", data_file, livetime
)

print(f"We see in 2615  keV {counts_2615_data} - {low_2615_data} + {high_2615_data} counts")
print(f"We see in 1764  keV {counts_1764_data} - {low_1764_data} + {high_1764_data} counts")
print(f"We see in 911  keV {counts_911_data} - {low_911_data} + {high_911_data} counts")
print(f"We see in 1461  keV {counts_1461_data} - {low_1461_data} + {high_1461_data} counts")


with PdfPages("plots/priors/prior_values.pdf") as pdf:
    for type in ["Bi212Tl208", "Pb214Bi214", "Ac228", "K40"]:
        for region in effs["all"].keys():
            if region == "Tl2615" and type != "Bi212Tl208":
                continue
            if region == "Bi1764" and type != "Pb214Bi214":
                continue
            if region == "Ac911" and type != "Ac228":
                continue
            if region == "K1461" and type != "K40":
                continue
            if region == "all" or region == "fit_range":
                continue

            labels = []
            categories = []
            y = []
            y_low = []
            y_high = []

            if region == "Tl2615":
                labels.append("Total")
                y.append(pred_2615)
                y_low.append(error_low_2615)
                y_high.append(error_high_2615)
                categories.append("total")
                data_band = (counts_2615_data, low_2615_data, high_2615_data)

            if region == "Bi1764":
                labels.append("Total")
                y.append(pred_1764)
                y_low.append(error_low_1764)
                y_high.append(error_high_1764)
                data_band = (counts_1764_data, low_1764_data, high_1764_data)
                categories.append("total")

            if region == "Ac911":
                labels.append("Total")
                y.append(pred_911)
                y_low.append(error_low_911)
                y_high.append(error_high_911)
                data_band = (counts_911_data, low_911_data, high_911_data)
                categories.append("total")
            if region == "K1461":
                labels.append("Total")
                y.append(pred_1461)
                y_low.append(error_low_1461)
                y_high.append(error_high_1461)
                data_band = (counts_1461_data, low_1461_data, high_1461_data)
                categories.append("total")

            for comp in priors["components"]:
                if type in comp["name"]:
                    labels.append(comp["name"])
                    eff = effs["all"][region][comp["name"]]
                    eff -= (
                        effs_left["all"][region][comp["name"]]
                        + effs_right["all"][region][comp["name"]]
                    ) / 2.0
                    y.append(eff * quantiles[comp["name"]][0])
                    y_low.append(eff * quantiles[comp["name"]][1])
                    y_high.append(eff * quantiles[comp["name"]][2])
                    categories.append(comp["type"])

            labels = utils.format_latex(np.array(labels))
            y = np.array(y)
            y_low = np.array(y_low)
            y_high = np.array(y_high)
            categories = np.array(categories)

            utils.make_error_bar_plot(
                np.arange(len(labels)),
                labels,
                y,
                np.abs(y_low),
                np.abs(y_high),
                obj=region,
                categories=categories,
                data_band=data_band,
            )

            plt.show()


lows = np.array(lows)
highs = np.array(highs)
pdf_high = utils.vals2hist(highs, copy.deepcopy(total_Tl))
for i in range(pdf_high.size - 2):
    pdf_high[i] += total_other[i]

### lets make a plot comparing the different shapes
with PdfPages("plots/priors/shapes.pdf") as pdf:
    utils.compare_mc(239j, pdfs, "Bi212Tl208", order, 2615j, pdf, 0, 4000, "log", colors=my_colors)
    utils.compare_mc(
        583j,
        pdfs,
        "Bi212Tl208",
        order,
        2615j,
        pdf,
        500,
        700,
        "linear",
        linewidth=0.8,
        colors=my_colors,
    )
    utils.compare_mc(
        727j,
        pdfs,
        "Bi212Tl208",
        order,
        2615j,
        pdf,
        700,
        900,
        "linear",
        linewidth=0.8,
        colors=my_colors,
    )
    utils.compare_mc(
        2104j, pdfs, "Bi212Tl208", order, 2615j, pdf, 1800, 2700, "linear", colors=my_colors
    )
    utils.compare_mc(100j, pdfs, "Pb214Bi214", order, 1764j, pdf, 0, 4000, "log", colors=my_colors)
    utils.compare_mc(
        609j,
        pdfs,
        "Pb214Bi214",
        order,
        1764j,
        pdf,
        500,
        1000,
        "linear",
        linewidth=0.8,
        colors=my_colors,
    )
    utils.compare_mc(
        1764j,
        pdfs,
        "Pb214Bi214",
        order,
        1764j,
        pdf,
        1000,
        1300,
        "linear",
        linewidth=0.8,
        colors=my_colors,
    )

    utils.compare_mc(
        1764j,
        pdfs,
        "Pb214Bi214",
        order,
        1764j,
        pdf,
        1600,
        2000,
        "linear",
        linewidth=0.8,
        colors=my_colors,
    )
    utils.compare_mc(
        1764j,
        pdfs,
        "Pb214Bi214",
        order,
        1764j,
        pdf,
        1300,
        1600,
        "linear",
        linewidth=0.8,
        colors=my_colors,
    )

    utils.compare_mc(
        2204j,
        pdfs,
        "Pb214Bi214",
        order,
        1764j,
        pdf,
        1900,
        2500,
        "linear",
        linewidth=0.8,
        colors=my_colors,
    )

    ### get rhe ratios


### also get the data
pdfs_tot = []
colors = []
labels = []
pdfs_tot.append(total)
colors.append("#000000")
labels.append("Total")

for comp in components:
    if comp == "2vbb" or comp == "alpha" or comp == "K42":
        pdfs_tot.append(hs[comp])
        colors.append(components[comp]["style"]["color"])
        labels.append(comp)
    if comp == "U":
        pdfs_tot.append(total_Bi)
        colors.append(components[comp]["style"]["color"])
        labels.append(comp)
    if comp == "Th":
        pdfs_tot.append(total_Tl)
        colors.append(components[comp]["style"]["color"])
        labels.append(comp)
        pdfs_tot.append(total_Ac)
        colors.append("blue")
        labels.append("Ac")
    if comp == "K40":
        pdfs_tot.append(total_K)
        colors.append(components[comp]["style"]["color"])
        labels.append(comp)


with PdfPages("plots/priors/total_contributions.pdf") as pdf:
    utils.plot_mc(total_Tl, "Total $^{212}$Bi+ $^{208}$Tl", pdf, data=data)
    utils.plot_mc(total_Bi, "Total $^{214}$Pb+ $^{214}$Bi", pdf, data=data)
    utils.plot_mc(total_Ac, "Total $^{228}$Ac", pdf, data=data)

    utils.plot_mc(total_other, "Total other", pdf, data=data)

    utils.plot_mc(total, "Total", pdf, data=data, pdf2=pdf_high)
    utils.plot_mc(
        total, "Total", pdf, data=data, range_x=(1900, 2700), range_y=(0.01, 500), pdf2=pdf_high
    )

    utils.plot_N_Mc(pdfs_tot, labels, "", pdf, data, (565, 4000), (0.05, 2e4), "log", colors)
    utils.plot_N_Mc(pdfs_tot, labels, "", pdf, data, (600, 1200), (0.05, 2000), "linear", colors)

    utils.plot_N_Mc(pdfs_tot, labels, "", pdf, data, (1300, 1600), (0.05, 10000), "linear", colors)

    utils.plot_N_Mc(pdfs_tot, labels, "", pdf, data, (1550, 3000), (0.05, 400), "linear", colors)
    utils.plot_N_Mc(pdfs_tot, labels, "", pdf, data, (1900, 2300), (0.05, 100), "linear", colors)


total_data = (
    utils.integrate_hist(data, 1930, 2099)
    + utils.integrate_hist(data, 2109, 2114)
    + utils.integrate_hist(data, 2124, 2190)
)
total = (
    utils.integrate_hist(total, 1930, 2099)
    + utils.integrate_hist(total, 2109, 2114)
    + utils.integrate_hist(total, 2124, 2190)
)
total_high = (
    utils.integrate_hist(pdf_high, 1930, 2099)
    + utils.integrate_hist(pdf_high, 2109, 2114)
    + utils.integrate_hist(pdf_high, 2124, 2190)
)

E = 2099 - 1930 + 2114 - 2109 + 2190 - 2124
M = 44
