"""
analyse-mult-two.py
author: Toby Dixon (toby.dixon.23@ucl.ac.uk)
Perform an analysis of multiplicity two data to see what we learn about bkg source location.
"""

from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
from matplotlib.patches import Polygon

from collections import OrderedDict
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
from hist import Hist
import hist
import argparse
import re
import utils
import json
from copy import copy
from matplotlib.backends.backend_pdf import PdfPages
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
cmap=tc.tol_cmap('iridescent')
cmap.set_under('w',1)
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))



style = {

    "yerr": False,
    "flow": None,
    "lw": 0.4,
}
data_path="../../l200a-vancouver23-dataset-v0.2.root"
with uproot.open(data_path) as f:
    histo=f["mul2_surv"]["all"].to_hist()


bin=5

### plot the 2D histogram 
### --------------------------------------------------------------------
    
histo_rebin =histo[hist.rebin(bin),hist.rebin(bin)]
w, x, y = histo_rebin.to_numpy()

a_region =np.array([[0,0],[0,1400],[500,1400-500],[500,500]])
b_region =np.array([[0,1400],[0,1500],[500,1500],[500,900]])
c_region =np.array([[0,1500],[0,6000],[500,6000],[500,1500]])
d_region =np.array([[500,500],[500,6000],[6000,6000]])

a_patch  = Polygon(a_region, closed=True, edgecolor='none', facecolor=vset.magenta, alpha=0.3,label="a ($E_{1}$)")
b_patch  = Polygon(b_region, closed=True, edgecolor='none', facecolor=vset.blue, alpha=0.3,label="b ($E_{1}+E_{2}$)")
c_patch  = Polygon(c_region, closed=True, edgecolor='none', facecolor=vset.orange, alpha=0.3,label="c ($E_1$)")
d_patch  = Polygon(d_region, closed=True, edgecolor='none', facecolor=vset.teal, alpha=0.3,label="d ($E_2$)")


my_cmap = copy(plt.cm.get_cmap('viridis'))
my_cmap.set_under('w')
fig, axes = lps.subplots(1, 1, figsize=(4,2), sharex=True, gridspec_kw = {'hspace': 0})
mesh = axes.pcolormesh(x, y, w.T, cmap=my_cmap,vmin=0.5)
plt.gca().add_patch(a_patch)
plt.gca().add_patch(b_patch)
plt.gca().add_patch(c_patch)
plt.gca().add_patch(d_patch)
plt.legend(loc="lower right",fontsize=6)
fig.colorbar(mesh)
axes.set_ylabel("Energy 1 [keV]")
axes.set_xlabel("Energy 2 [keV]")
axes.set_xlim(0,1500)
axes.set_ylim(0,3000)
#plt.show()
#plt.savefig("plots/mult_two/2D_spec.png")



### extract the 1D histograms 
### ------------------------------------------------------------------------



fig_2, axes_2 = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
hist_energy_0=histo.project(0)[hist.rebin(bin)]
hist_energy_1=histo.project(1)[hist.rebin(bin)]
hist_sum = utils.project_sum(histo)[hist.rebin(bin)]


hist_sum.plot(ax=axes_2,color=vset.orange,**style,label="Summed energy ")

hist_energy_0.plot(ax=axes_2,color=vset.cyan,**style,label="Energy 2")


hist_energy_1.plot(ax=axes_2,color=vset.blue,**style,label="Energy 1")
axes_2.set_yscale("linear")
axes_2.set_xlim(0,4000)
axes_2.set_xlabel("Energy [keV]")
axes_2.set_ylabel("counts/ {} keV".format(bin))
legend=axes_2.legend(loc='upper right',edgecolor="black",frameon=False, facecolor='white',framealpha=1)

plt.show()

### get some counting rates
### ------------------------------------------------------------------------

peaks=[511,583,609,1525,2615]
for peak in peaks:
    energies=np.array([peak-6,peak-2,peak+2,peak+6])
    counts_energy_1=np.array([utils.integrate_hist(hist_energy_1,energies[0],energies[1]),
                    utils.integrate_hist(hist_energy_1,energies[1],energies[2]),
                    utils.integrate_hist(hist_energy_1,energies[2],energies[3])
                    ])
    counts_energy_2=np.array([utils.integrate_hist(hist_energy_0,energies[0],energies[1]),
                    utils.integrate_hist(hist_energy_0,energies[1],energies[2]),
                    utils.integrate_hist(hist_energy_0,energies[2],energies[3])
                    ])
    
    N_1,el_1,eh_1=utils.get_counts_minuit(counts_energy_1,energies)
    N_2,el_2,eh_2=utils.get_counts_minuit(counts_energy_2,energies)

    print("for peak at {} [energy 1]  {:0.1f} ^+{:0.1f} _-{:0.1f}".format(peak,N_1,eh_1,el_1))
    print("for peak at {} [energy 2]  {:0.1f} ^+{:0.1f} _-{:0.1f}".format(peak,N_2,eh_2,el_2))