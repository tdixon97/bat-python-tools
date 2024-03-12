from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')

from collections import OrderedDict
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
import hist
import argparse
import re
import utils
import json
from hist import Hist
import os
import numpy as np
from IPython.display import display
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.6,
}



cfg ="../hmixfit/inputs/cfg/vancouver/workshop_fits/l200a-vancouver_workshop-v3.0_2d.json"
with open(cfg,"r") as json_file:
    cfg_dict =json.load(json_file)

### extract all we need from the cfg dict
fit_name,out_dir,det_types,ranges,dataset_name=utils.parse_cfg(cfg_dict)
outfile = out_dir+"/hmixfit-"+fit_name+"/histograms.root"
ds=["{}_{}".format(dataset_name,det_types[4].split("/")[1])]
print(ds)

with uproot.open(outfile) as f:
    hist=f[ds[0]]["total_model"].to_hist()


w,x,y= hist.to_numpy()
bin_edges=[]
vals=[]
xb =np.diff(x)
yb=np.diff(y)
for i in range(len(x)-1):
    for j in range(len(y)-1):
        bin_edges.append(x[i]+(y[j]/3000)*(x[i+1]-x[i]))
        vals.append(w[i,j])
bin_edges.append(x[-1])

histo =( Hist.new.Variable(bin_edges).Double())
print(len(vals))
print(len(bin_edges))
width=np.diff(bin_edges)
print(width)
for i in range(histo.size-3):
  
    histo[i]=vals[i]/width[i]
fig, axes_full = lps.subplots(1, 1, figsize=(6,2), sharex=True)

histo.plot(ax=axes_full,**style,color=vset.blue)
axes_full.set_xlabel("Flattened Energy [keV]")
axes_full.set_ylabel("Counts/keV$^{2}$")
plt.show()