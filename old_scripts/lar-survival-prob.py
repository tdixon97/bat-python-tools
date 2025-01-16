"""
lar-survival-prob.py
Author: Toby Dixon
Date  : 26/11/2023
This script is meant to study the survival probability in different regions for the LAR cut.
Both in the data, and different MC contributions.
"""

from __future__ import annotations

from legend_plot_style import LEGENDPlotStyle as lps

lps.use("legend")

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
import uproot
import utils

vset = tc.tol_cset("vibrant")
mset = tc.tol_cset("muted")
plt.rc("axes", prop_cycle=plt.cycler("color", list(vset)))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}

parser = argparse.ArgumentParser(description="A script with command-line argument.")
parser.add_argument("-p", "--peak", type=int, help="peak energy", default=2615)
parser.add_argument("-n", "--name", type=str, help="peak name", default="Tl_peak")

args = parser.parse_args()
peak = args.peak
name_peak = args.name
data_path = "../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v0.1.root"
pdf_path = "../hmixfit/inputs/pdfs/l200a-pdfs_vancouver23/"
cfg_path = "../hmixfit/inputs/cfg/vancouver/l200a-vancouver-m1-m2.json"
file = uproot.open(data_path)
with open(cfg_path) as file_cfg:
    cfg = json.load(file_cfg)

regions = {name_peak: (peak - 5, peak + 5)}
left_sideband = {name_peak: (peak - 15, peak - 5)}
right_sideband = {name_peak: (peak + 5, peak + 15)}

det_types, name = utils.get_det_types("sum")


energies = np.array(
    [
        left_sideband[name_peak][0],
        left_sideband[name_peak][1],
        regions[name_peak][1],
        right_sideband[name_peak][1],
    ]
)

data_counts = utils.get_data_counts_total(
    "mul_surv", det_types, regions, file, key_list=[name_peak]
)
data_counts_right = utils.get_data_counts_total(
    "mul_surv", det_types, right_sideband, file, key_list=[name_peak]
)
data_counts_left = utils.get_data_counts_total(
    "mul_surv", det_types, left_sideband, file, key_list=[name_peak]
)

data_counts_lar = utils.get_data_counts_total(
    "mul_lar_surv", det_types, regions, file, key_list=[name_peak]
)
data_counts_lar_right = utils.get_data_counts_total(
    "mul_lar_surv", det_types, right_sideband, file, key_list=[name_peak]
)
data_counts_lar_left = utils.get_data_counts_total(
    "mul_lar_surv", det_types, left_sideband, file, key_list=[name_peak]
)


### the efficiency calculation
eff, eff_low, eff_high = utils.get_eff_minuit(
    np.array(
        [
            data_counts_left["all"][name_peak],
            data_counts["all"][name_peak],
            data_counts_right["all"][name_peak],
        ]
    ),
    np.array(
        [
            data_counts_lar_left["all"][name_peak],
            data_counts_lar["all"][name_peak],
            data_counts_lar_right["all"][name_peak],
        ]
    ),
    energies,
)
eff *= 100
eff_low *= 100
eff_high *= 100

### now we have data efficiency get some for MC
effs = utils.get_total_efficiency(det_types, cfg, "mul_surv", regions, pdf_path)
effs_lar = utils.get_total_efficiency(det_types, cfg, "mul_lar_surv", regions, pdf_path)


### now loop over the keys
eff_list = []
names = []
eff_map = effs["all"][name_peak]
eff_map_lar = effs_lar["all"][name_peak]

for key in sorted(effs["all"][name_peak].keys()):
    if (name_peak == "Tl_peak" and "Tl208" in key) or (name_peak == "Bi_peak" and "Bi214" in key):
        eff_tmp = eff_map[key]
        eff_tmp_lar = eff_map_lar[key]
        eff_list.append(100 * eff_tmp_lar / eff_tmp)

        names.append(key[11:])
    elif name_peak == "K_peak" and "K" in key:
        eff_tmp = eff_map[key]
        eff_tmp_lar = eff_map_lar[key]
        if eff_tmp > 0:
            eff_list.append(100 * eff_tmp_lar / eff_tmp)

            names.append(key[3:])


fig, axes = lps.subplots(1, 1, figsize=(5, 4), sharex=True, gridspec_kw={"hspace": 0})

axes.set_title(f"LAr survival for {name_peak} ( {peak} keV)")
axes.errorbar(eff_list, names, color=vset.blue, fmt="o", label="MC")
axes.set_xlabel("LAr survival probability [%]")
axes.set_yticks(np.arange(len(names)), names, rotation=0, fontsize=8)
axes.axvline(x=eff, color=vset.red, label="data")
axes.axvspan(xmin=eff - eff_low, xmax=eff + eff_high, alpha=0.2, color=vset.orange)
axes.legend(loc="best", edgecolor="black", frameon=True, facecolor="white", framealpha=1)
plt.savefig(f"plots/lar/effs_{peak}.pdf")
