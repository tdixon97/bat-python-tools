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

vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))
# mpl.rcParams["font.size"] = 8


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.8,
}

components = OrderedDict({
    "2vbb": {
        "hists": ["Nvbb_bege", "Nvbb_icpc", "Nvbb_coax", "Nvbb_ppc"],
        "style": {
            "color": mset.olive,
            "label": r"$2\nu\beta\beta$",
        }
    },
    "Th": {
        "hists": [
            'Ac228_pen_plates',
            'Bi212Tl208_birds_nest',
            'Bi212Tl208_cables',
            'Bi212Tl208_fiber_shroud',
            'Bi212Tl208_front_end_electronics',
            'Bi212Tl208_hpge_support_copper',
            'Bi212Tl208_mini_shroud',
            'Bi212Tl208_pen_plates',
            'Bi212Tl208_wls_reflector',
        ],
        "style": {
            "color": vset.cyan,
            "label": "Thorium",
        }
    },
    "U": {
        "hists": [
            'Pb214Bi214_birds_nest',
            'Pb214Bi214_cables',
            'Pb214Bi214_fiber_shroud',
            'Pb214Bi214_front_end_electronics',
            'Pb214Bi214_hpge_support_copper',
            'Pb214Bi214_mini_shroud',
            'Pb214Bi214_pen_plates',
            'Pb214Bi214_wls_reflector',
        ],
        "style": {
            "color": vset.teal,
            "label": "Uranium",
        }
    },
    "K40": {
        "hists": [
            'K40_front_end_electronics',
            'K40_hpge_support_copper',
            'K40_pen_plates',
            'K40_wls_reflector',
        ],
        "style": {
            "color": vset.red,
            "label": "$^{40}$K"
        }
    },
    "K42": {
        "hists": [
            'K42_homogeneous',
            'K42_hpge_shell',
            'K42_hpge_surface',
            'K42_mini_shroud',
        ],
        "style": {
            "color": vset.magenta,
            "label": "$^{42}$K"
        }
    },
    "total_model": {
        "hists": ["total_model"],
        "style": {
            "label": "Model",
            "color": vset.black,
        }
    },
    "data": {
        "hists": ["fitted_data"],
        "style": {
            "label": "Data",
            # "color": vset.black,
            # "histtype": "errorbar",
            # "markersize": 3,
            #"linewidth": 0,
            "histtype": "fill",
            "color": vset.blue,
            "alpha": 0.25,
        }
    },
})
parser = argparse.ArgumentParser(description='A script with command-line argument.')
parser.add_argument('-d','--det_type', type=str, help='Detector type',default='all')
parser.add_argument('-l','--lower_x', type=int, help='Low range',default=565)
parser.add_argument('-u','--upper_x', type=int, help='Upper range',default=3000)
parser.add_argument("-s","--scale",type=str,help="Scale",default="log")
parser.add_argument("-o","--out_file",type=str,help="file",default="../results/hmixfit-l200a-taup-silver-dataset-m1-histograms.root")


args = parser.parse_args()
det_type=args.det_type
xlow=args.lower_x
xhigh=args.upper_x
ylowlim =0.05
scale=args.scale
outfile=args.out_file
def get_hist(obj):
    return obj.to_hist()[552:2982][hist.rebin(15)]

if det_type=="all":
    datasets = ["l200a_taup_silver_dataset_bege", "l200a_taup_silver_dataset_icpc","l200a_taup_silver_dataset_ppc","l200a_taup_silver_dataset_coax"]
    labels = ["BEGe detectors after QC", "ICPC detectors after QC","PPC detectors after QC","COAX detectors after QC"]
else:
    datasets=["l200a_taup_silver_dataset_{}".format(det_type)]
    labels=["{} detectors after QC".format(det_type)]
print(datasets)
y=6
if det_type!="all":
    y=4

if (det_type=="all"):
    fig, axes = lps.subplots(len(datasets), 1, figsize=(6, y), sharex=True, gridspec_kw = {'hspace': 0})
else:
    fig, axes_full = lps.subplots(len(datasets)+1, 1, figsize=(6, y), sharex=True, gridspec_kw = { 'height_ratios': [8, 2],"hspace":0})

if det_type=="all":
    
    for ax in axes:
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", direction="in")
else:
    axes=np.array([axes_full[0]])
    
with uproot.open(outfile) as f:
    for ds, ax, title in zip(datasets, axes.flatten(), labels):
        for comp, info in components.items():
            h = None
            for name in info["hists"]:
                if name not in f[ds]["originals"]:
                    continue
                if h is None:
                    h = get_hist(f[ds]["originals"][name])
                else:
                    h += get_hist(f[ds]["originals"][name])
            for i in range(h.size - 2):
                if h[i] == 0:
                    h[i] = 1.05*ylowlim

            h.plot(ax=ax, **style, **info["style"])
            if (comp=="data"):
                data=h.values()
                bins=h.axes.centers[0]
            if (comp=="total_model"):
                pred=h.values()
        residual = np.array([((d - m) / (m ** 0.5)) if d > -1 else 0 for d, m in zip(data, pred)])

        if ds == datasets[0]:
            ax.legend(loc="upper right", ncols=3)
            ax.set_legend_annotation()
        elif ds==datasets[len(datasets)-1]:
            ax.xaxis.set_tick_params(top=False)
            ax.set_legend_logo(position='upper right', logo_type = 'preliminary', scaling_factor=7)
        else:
            ax.xaxis.set_tick_params(top=False)
            ax.annotate(title, (0.02, 0.88), xycoords="axes fraction", fontsize=8)

        
        masked_values = np.ma.masked_where((bins > xhigh)  | (bins < xlow), pred)
        masked_values_data = np.ma.masked_where((bins > xhigh)  | (bins < xlow), data)

        maxi = max(masked_values.max(),masked_values_data.max())
        max_y = maxi+2*np.sqrt(maxi)+1
        ax.set_yscale("linear")
        ax.set_ylim(bottom=ylowlim,top=max_y)      
        #ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts / 15 keV")
        ax.set_yscale(scale)
        ax.set_xlim(xlow,xhigh)
        
        ### now plot the residual
        if (det_type!="all"):
            
            axes_full[1].errorbar(bins,residual,yerr=np.ones(len(residual)),fmt="o",color=vset.blue,markersize=2)
            axes_full[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
            axes_full[1].set_xlabel("Energy (keV)")
            axes_full[1].set_ylabel("Residual")
            axes_full[1].set_yscale("linear")
            axes_full[1].set_xlim(xlow,xhigh)
            axes_full[1].set_ylim(-5,5)
            axes_full[1].xaxis.set_tick_params(top=False)

            plt.tight_layout()

#plt.subplots_adjust(hspace=0)
plt.tight_layout()

plt.savefig("l200a-taup-bkg-model_{}_{}_{}_to_{}.pdf".format(det_type,scale,xlow,xhigh))
#plt.savefig("l200a-taup-bkg-model.png")
#plt.show()


