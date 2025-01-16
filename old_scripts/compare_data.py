from __future__ import annotations

from legend_plot_style import LEGENDPlotStyle as lps

lps.use("legend")

from copy import deepcopy

import matplotlib.pyplot as plt
import tol_colors as tc
import uproot
import utils

style = {
    "yerr": False,
    "flow": None,
    "lw": 0.6,
}
vset = tc.tol_cset("vibrant")
mset = tc.tol_cset("muted")
plt.rc("axes", prop_cycle=plt.cycler("color", list(vset)))
path_1 = "~/Downloads/vancouver_dataset_20240303_refv1skm.root"
path_2 = "~/Downloads/l200a-vancouver23-dataset-v1.0.root"

hist_1 = "spectrum"
hist_2 = "mul_surv/all"

label_1 = "Sofiia spectrum (skm files)"
label_2 = "Tobiia spectrum (evt files)"

with uproot.open(path_1) as f1:
    h1 = utils.get_hist(f1[hist_1], (0, 3000), 1)


with uproot.open(path_2) as f2:
    print(f2[hist_2].to_hist()[0:3000])
    h2 = utils.get_hist(f2[hist_2], (0, 3000), 1)


fig, axes_full = lps.subplots(
    2, 1, figsize=(6, 4), sharex=True, gridspec_kw={"height_ratios": [8, 2], "hspace": 0}
)

hdiff = deepcopy(h1)

for i in range(h1.size - 2):
    hdiff[i] -= h2[i]
    h2[i] = h2[i] / 1


h1.plot(ax=axes_full[0], **style, label=label_1, color=vset.blue)
h2.plot(ax=axes_full[0], **style, label=label_2, color=vset.orange)
axes_full[1].set_xlabel("Energy [keV]")
axes_full[0].set_ylabel("counts/5 keV")
axes_full[0].set_yscale("log")
axes_full[0].set_xlim(0, 3000)

axes_full[0].legend(loc="upper right")

axes_full[1].scatter(hdiff.axes.centers[0], hdiff.values(), color="black", linewidth=0)
axes_full[1].set_ylabel("Difference")
plt.tight_layout()
plt.show()
