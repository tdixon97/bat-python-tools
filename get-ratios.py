"""
get-ratios.py
author: Toby Dixon (toby.dixon.23@ucl.ac.uk)

"""

from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
from tqdm import tqdm
from collections import OrderedDict
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
from hist import Hist
import hist
import os
import sys
import argparse
import re
import utils
import variable_binning
import json
from copy import copy
from copy import deepcopy
from matplotlib.backends.backend_pdf import PdfPages


vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
lset = tc.tol_cset('light')

cmap=tc.tol_cmap('iridescent')
cmap.set_under('w',1)
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))
 

def get_mc_data(priors,ratios_data,ratios,cat,spec,peak):
    data_val=ratios_data[cat][spec][peak]

    ## get mc
    ratio_tmp={}
          
    for comp in priors["components"]:
        if (spec in ratios[comp["name"]][cat]) and (peak in ratios[comp["name"]][cat][spec]):
            ratio_tmp[comp["name"][11:]]=ratios[comp["name"]][cat][spec][peak]
            if (("Po214") in comp["name"]) or ("Pb214" in comp["name"]):
                denom="N(1764, M1)"
            else:
                denom ="N(2615, M1)"

            if (spec=="mul_surv"):
                spec_name="M1"
            elif (spec=="mul2_e1"):
                spec_name="M2-E1"
            elif (spec=="mul2_e2"):
                spec_name="M2-E2"
            elif (spec=="mul2_sum"):
                spec_name="M2-Sum"
            elif "mul2_surv_e2" in spec:
                spec_name=" , "+spec.split("e2_")[1]+"M2"
            else:
                spec_name=spec
    return ratio_tmp,data_val,spec_name,denom


def quantiles_to_sym_error(line,factor=100):
    """Get the med and error from the quantiles (TODO: replace with assymetric error)"""

    
    med = (line["q16"]+line["q84"])/2.
    err = med-line["q16"]
 
    if (("q90" in line) and err>line["q16"]):
        med=0
        error=line["q90"]
    else:
        error = med-line["q16"]
    
    return (factor*med,factor*error)

def project_gamma_line(histo,gamma_list=np.array([]),low_energy=500,high_energy=1500,peak_energy=(609,3),variable_select="e2",variable_project="e1",
                       title="Projection of $E_2$ 609 keV",counting_rates={},name="m2_project",plot=False,just_value=False,name_comb=""):
    """ Produce the plots projecting the gamma lines"""
    gamma_sig=gamma_list
    
    gamma_bkg =np.array([1461-peak_energy[0],1525-peak_energy[0]])
    if (peak_energy[0]<1461):
        gamma_list=np.append(gamma_list,gamma_bkg)
    gamma_list=np.sort(gamma_list)

    bin_edges_1 =variable_binning.compute_binning(gamma_energies=gamma_list,
                                                  low_energy=low_energy,high_energy=high_energy,
                                                  gamma_binning_low=10,gamma_binning_high=10,cont_binning=50)

    if (variable_project=="e1"):
        idx=1
    else:
        idx=0
    histo_projection_1 = utils.select_region(histo,
                                             [{"var":variable_select,"greater":False,"val":peak_energy[0]+peak_energy[1]},
                                              {"var":variable_select,"greater":True,"val":peak_energy[0]-peak_energy[1]}]).project(idx)

    counting_rates["all"][name]={}
    ##### get the counts
    total =0
    error_total=0
    for p in gamma_sig:
        if p>high_energy or p<low_energy:
            continue
        counts =np.array([utils.integrate_hist(histo_projection_1,p-30,p-5),
                            utils.integrate_hist(histo_projection_1,p-5,p+5),
                            utils.integrate_hist(histo_projection_1,p+5,p+30),
                            ],dtype="int")
        energy = np.array([p-30,p-5,p+5,p+30])
          
        signal,elow,ehigh = utils.get_counts_minuit(counts,energy)
        error=(elow+ehigh)/2.
        if (just_value==False):
            counting_rates["all"][name][str(p)]=(signal,error)
            
        else:
            counting_rates["all"][name][str(p)]=signal
        total+=signal
        error_total+=error*error
    
    if (just_value==False):
        counting_rates["all"][name][name_comb]=(total,np.sqrt(error_total))
            
    else:
        counting_rates["all"][name][name_comb]=total

    ## now rebin
    if (plot):
        histo_projection_1=utils.variable_rebin(histo_projection_1,bin_edges_1)
        histo_projection_1=utils.normalise_histo(histo_projection_1,10)
        
        fig, axes = lps.subplots(1, 1, figsize=(5,3), sharex=True, gridspec_kw = {'hspace': 0})
        histo_projection_1.plot(ax=axes,color=vset.blue,**style,histtype="fill",alpha=0.4)

        axes.set_title(title)
        if (variable_project=="e1"):        
            axes.set_xlabel("Energy 1 [keV]")
        else:
            axes.set_xlabel("Energy 2 [keV]")

        axes.set_ylabel("counts/10 keV")
        max =axes.get_ylim()[1]+2
        axes.set_xlim(low_energy,high_energy)
        axes.vlines(x=gamma_sig,color=vset.teal,ymin=0,ymax=max,label="Signal",linewidth=0.5,linestyle="--")
        if (peak_energy[0]<1525):
            axes.vlines(x=gamma_bkg,color=vset.red,ymin=0,ymax=max,label="K background",linewidth=0.5,linestyle="--")
        axes.set_ylim(0,max)
        plt.legend(loc="best",edgecolor="black",frameon=True, facecolor='white',framealpha=1,fontsize=8)

def main():

    def get_ratio(a,b):
        """ Basic error propogation on the ratio"""
        ### note this is quite wrong if a or b are small 
        if (a[0]>0 and b[0]>0):
            r=100*a[0]/b[0]
            return r, np.sqrt(r*r*b[1]*b[1]/(b[0]*b[0])+r*r*a[1]*a[1]/(a[0]*a[0]))
        elif (b[0]==0):
            return 0,100

        elif (a[0]==0):
            return 0,np.sqrt(100*100*a[1]*a[1]/(b[0]*b[0]))
        
    style = {

        "yerr": False,
        "flow": None,
        "lw": 0.2,
    }

    data_path="../hmixfit/inputs/data/datasets/l200a-vancouver23-dataset-v1.0.0.root"
    pdf_path = "../hmixfit/inputs/pdfs/l200a-pdfs_vancouver23_v6.0/l200a-pdfs.0/"
    cats=["cat_1","cat_2","cat_3"] #,"sd_0","sd_1","sd_2","sd_3","sd_4","sd_5"]
    m2_cats=["all","cat_1","cat_2","cat_3"] #,"sd_0","sd_1","sd_2","sd_3","sd_4","sd_5"]

    groups=["icpc","bege","ppc","coax"] #,"top","mid_top","mid","mid_bottom","bottom"]
    histo_cats=[]
    histo_groups={}
    with uproot.open(data_path,object_cache=None) as f:
        histo=f["mul2_surv_2d"]["all"].to_hist()
        histo_m1=f["mul_surv"]["all"].to_hist()

        for cat in cats:
            histo_cats.append(f["mul2_surv_2d"][cat].to_hist())

        for group in groups:
            histo_groups[group]=f["mul_surv"][group].to_hist()
    
    histo_groups["all"]=histo_m1

   
    ### some parameters of what to do
    make_plots=False
    bin=5
    plot_regions=False
    make_2D_plots=False
    plot_data_1D=False
    plot_projections=False
    plot_rebin_2D=False
    plot_categories_1D=False
    plot_mc=False
    get_from_data=False

    gamma_line_M1_path = "gamma_line_output/intensities_ratios_20242501.json"
    gamma_line_Bi_path = "gamma_line_output/intensities_ratios_20240602.json"

    gamma_line_groups_path = "gamma_line_output/intensities_fits_by_group_20240128.json"
    gamma_line_M2_path = "gamma_line_output/intensities_ratios_M2_20242501_v2.json"
    gamma_line_3MeV_M1_path = "gamma_line_output/intensities_ratios_M1_3MeV_20242601.json"
    gamma_line_3MeV_M2_path = "gamma_line_output/intensities_ratios_M2_3MeV_20242601.json"

    gamma_line_M2_cats = "gamma_line_output/intensities_ratios_M2_cats_20242502_3.json"

    if (get_from_data):
        save_data="out/data_counts.json"
        save_data_ratio="out/data_ratios.json"
    else:
        save_data="out/data_counts_gla.json"
        save_data_ratio="out/data_ratios_gla.json"

    save_mc="out/mc_counts.json"
    save_mc_ratio="out/mc_ratios.json"
    compute_mc=False


    ### plot the 2D histogram 
    ### --------------------------------------------------------------------
    if (plot_categories_1D):
        histo_rebin =histo[hist.rebin(bin),hist.rebin(bin)]
        w, x, y = histo_rebin.to_numpy()
        histo_a = utils.select_region(histo,[{"var":"e2","greater":False,"val":500},{"var":"sum","greater":False,"val":1400}])
        histo_b = utils.select_region(histo,[{"var":"e2","greater":False,"val":500},{"var":"sum","greater":True,"val":1400},{"var":"e1","greater":False,"val":1500}])
        histo_c = utils.select_region(histo,[{"var":"e2","greater":False,"val":500},{"var":"e1","greater":True,"val":1500}])
        histo_d = utils.select_region(histo,[{"var":"e2","greater":True,"val":500}])

        histos=[histo,histo_a,histo_b,histo_c,histo_d,histo_cats[0],histo_cats[1],histo_cats[2]]
        names=["full","a","b","c","d","cat_1","cat_2","cat_3"]

    if (make_2D_plots):
        
        #### plot the 2D histo (regions)
        ### ---------------------------------------------------------------

        for h,n in zip(histos,names):
            histo_rebin =h[hist.rebin(bin),hist.rebin(bin)]
            w, x, y = histo_rebin.to_numpy()

            a_region =np.array([[0,0],[0,1400],[500,1400-500],[500,500]])
            b_region =np.array([[0,1400],[0,1500],[500,1500],[500,900]])
            c_region =np.array([[0,1500],[0,6000],[500,6000],[500,1500]])
            d_region =np.array([[500,500],[500,6000],[6000,6000]])

            a_patch  = Polygon(a_region, closed=True, edgecolor='none', facecolor=vset.magenta, alpha=0.2,label="a ($E_{1}$)")
            b_patch  = Polygon(b_region, closed=True, edgecolor='none', facecolor=vset.blue, alpha=0.2,label="b ($E_{1}+E_{2}$)")
            c_patch  = Polygon(c_region, closed=True, edgecolor='none', facecolor=vset.orange, alpha=0.2,label="c ($E_1$)")
            d_patch  = Polygon(d_region, closed=True, edgecolor='none', facecolor=vset.teal, alpha=0.2,label="d ($E_2$)")
            
            w, x, y = histo_rebin.to_numpy()

            my_cmap = copy(plt.cm.get_cmap('viridis'))
            my_cmap.set_under('w')
            fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
            mesh = axes.pcolormesh(x, y, w.T, cmap=my_cmap,vmin=0.5)

            ## shade some regions
            if plot_regions:
                plt.gca().add_patch(a_patch)
                plt.gca().add_patch(b_patch)
                plt.gca().add_patch(c_patch)
                plt.gca().add_patch(d_patch)

            plt.legend(loc="lower right",fontsize=6)
            fig.colorbar(mesh)
            axes.set_ylabel("Energy 1 [keV]")
            axes.set_xlabel("Energy 2 [keV]")
            axes.set_xlim(0,1200)
            plt.title(n)
            axes.set_ylim(0,3000)
            if plot_regions==False:
                plt.savefig("plots/mult_two/2D_spec_{}.png".format(n))
            else:
                plt.savefig("plots/mult_two/2D_spec_{}_plus_regions.png".format(n))
            plt.show()


    ### extract the 1D histograms (data) and get the counting rates
    ### ------------------------------------------------------------------------

    gamma_list_609 =np.array([768,934, 1120, 1238, 1408])
    gamma_list_583 = np.array([277,511,763,2615])
    gamma_list_2615 = np.array([277,511,583,763,860])
    gamma_list_911 = np.array([463])
    gamma_list_Tl = np.concatenate([gamma_list_583,gamma_list_2615,np.array([3197,3125])])


    ### either get directly from data or read from the gamma line analysis file
    if (get_from_data):
        fig_2, axes_2 = lps.subplots(1, 1, figsize=(4,3), sharex=True, gridspec_kw = {'hspace': 0})
        hist_energy_0=histo.project(0)[hist.rebin(bin)]
        hist_energy_1=histo.project(1)[hist.rebin(bin)]
        hist_sum = utils.project_sum(histo)[hist.rebin(bin)]

        if (plot_data_1D):
            hist_sum.plot(ax=axes_2,color=vset.orange,**style,label="$E_{1}+E_{2}$ ")
            hist_energy_0.plot(ax=axes_2,color=vset.cyan,**style,label="$E_{2}$")
            hist_energy_1.plot(ax=axes_2,color=vset.blue,**style,label="$E_{1}$")
            axes_2.set_yscale("log")
            axes_2.set_xlim(0,4000)

            axes_2.set_xlabel("Energy [keV]")
            axes_2.set_ylabel("counts/ {} keV".format(bin))
            legend=axes_2.legend(loc='upper right',edgecolor="black",frameon=False, facecolor='white',framealpha=1)
            plt.show()

        ##### get the counting rates
        ####  -------------------------------------------------------------------------------------------

        data_counts={}
        histos=[histo]
        histos.extend(histo_cats)
        names=["all","cat_1","cat_2","cat_3"] #"sd_0","sd_1","sd_2","sd_3","sd_4","sd_5"]
        names.extend(groups)
        for i in range(len(groups)):
            histos.append(None)
        spectra = ["mul_surv","mul2_e1","mul2_e2","mul2_sum"]
        ranges=[(0,4000),(0,3000),(0,1500),(0,4000)]

        peaks=[583,609,1764,2615,3197,2204,3125,1729,1847,1377,2358,1543,2017,1274,1415]
        for h,n in zip(histos,names):
            data_counts[n]={}
            for s in spectra:
                if s=="mul_surv":
                    if (n not in ["all","icpc","bege","coax","ppc"]):
                        continue
                    h1D= histo_groups[n]
                elif s=="mul2_e1":
                    if (n not in m2_cats):
                        continue
                    h1D = h.project(1)
                elif s=="mul2_e2":
                    if (n not in m2_cats):
                        continue
                    h1D = h.project(0)
                else:
                    if (n not in m2_cats):
                        continue
                    h1D = utils.project_sum(h)
                data_counts[n][s]={}

                for p in peaks:
                    size=5
                    if p<800:
                        size=3
                    counts =np.array([utils.integrate_hist(h1D,p-20,p-size),
                                    utils.integrate_hist(h1D,p-size,p+size),
                                    utils.integrate_hist(h1D,p+size,p+20),
                                    ],dtype="int")
                    
                    energy = np.array([p-20,p-size,p+size,p+20])
                    if (s=="mul2_e2" and (p==2615 or p==1764)):
                        continue
                    signal,elow,ehigh = utils.get_counts_minuit(counts,energy)
                    error=(elow+ehigh)/2.
                    data_counts[n][s][str(p)]=(signal,error)



        #### now do some projections of known gamma lines
        ### ----------------------------------------------
        ### 1) E1 for E2=609+/-3 keV
        ### 2) E1 for E2=583+/-3 keV
        ### 3) E2 for E1=2615+/-5 keV



        project_gamma_line(histo,gamma_list_609,500,1500,(609,3),"e2","e1","Projection of $E_2$ 609 keV",data_counts,"mul2_surv_e2_609",name_comb="Bi")
        #project_gamma_line(histo,gamma_list_609,0,600,(609,3),"e1","e2","Projection of $E_1$ 609 keV",data_counts,"mul2_surv_e1_609")

        project_gamma_line(histo,gamma_list_583,500,2700,(583,3),"e2","e1","Projection of $E_2$ 583 keV",data_counts,"mul2_surv_e2_583",name_comb="Tl_583_E2")
        project_gamma_line(histo,gamma_list_583,0,600,(583,3),"e1","e2","Projection of $E_1$ 583 keV",data_counts,"mul2_surv_e1_583",name_comb="Tl_583_E1")

        project_gamma_line(histo,gamma_list_2615,0,1000,(2615,5),"e1","e2","Projection of $E_1$ 2615 keV",data_counts,"mul2_surv_e1_2615",name_comb="Tl_2615_E1")

        project_gamma_line(histo,gamma_list_911,0,1000,(911,3),"e1","e2","Projection of $E_1$ 911 keV",data_counts,"mul2_surv_e1_911",name_comb="Ac")



        ratios_data=deepcopy(data_counts)


        for cat in data_counts:
            for spec in data_counts[cat]:
                for peak in data_counts[cat][spec]:
                    if (np.isin(peak,gamma_list_Tl)):
                        ratios_data[cat][spec][peak]=get_ratio(data_counts[cat][spec][peak],data_counts["all"]["mul_surv"]["2615"])
                    else:
                        ratios_data[cat][spec][peak]=get_ratio(data_counts[cat][spec][peak],data_counts["all"]["mul_surv"]["1764"])

        sum_3125_3197  = data_counts["all"]["mul_surv"]["3125"][0]+data_counts["all"]["mul_surv"]["3197"][0]
        sum_err_3125_3197 = np.sqrt(np.power(data_counts["all"]["mul_surv"]["3125"][0],2)+
                                    np.power(data_counts["all"]["mul_surv"]["3197"][1],2))

        bi_sum_lines=["1377","1543","1729","1847","2017","1274","1415"]
        bi_sum =0
        bi_sum_err=0
        for bi in bi_sum_lines:
            bi_sum +=data_counts["all"]["mul_surv"][bi][0]
            bi_sum_err+=np.power(data_counts["all"]["mul_surv"][bi][0],2)
        
        bi_sum_err=np.sqrt(bi_sum_err)

        ratios_data["all"]["mul_surv"]["bi_sum"]=get_ratio((bi_sum,bi_sum_err),data_counts["all"]["mul_surv"]["1764"])                                       

        ratios_data["all"]["mul_surv"][str(3125+3197)]=get_ratio((sum_3125_3197,sum_err_3125_3197),data_counts["all"]["mul_surv"]["2615"])                                       
        
        with open(save_data, "w") as json_file:
            json.dump(data_counts, json_file,indent=2)

        with open(save_data_ratio, "w") as json_file:
            json.dump(ratios_data, json_file,indent=2)

    #### get from the gamma line analysis
    #### --------------------------------
    else:
        ratios_data={"all":{"mul_surv":{},"mul2_sum":{},"mul2_e1":{},"mul2_e2":{}}}

        for cat in m2_cats:
      
            
            ratios_data[cat]={"mul_surv":{},"mul2_sum":{},"mul2_e1":{},"mul2_e2":{}}
       

        data_counts={"all":{"mul_surv":{},"mul2_sum":{},"mul2_e1":{},"mul2_e2":{}}}

        with open(gamma_line_M1_path, "r") as json_file:
            gamma_line_M1 = json.load(json_file)

        with open(gamma_line_Bi_path, "r") as json_file:
            gamma_line_Bi = json.load(json_file)

        with open(gamma_line_groups_path, "r") as json_file:
            gamma_line_groups = json.load(json_file)

        with open(gamma_line_M2_path, "r") as json_file:
            gamma_line_M2 = json.load(json_file)

        with open(gamma_line_3MeV_M1_path, "r") as json_file:
            gamma_line_3MeV_M1 = json.load(json_file)

        with open(gamma_line_3MeV_M2_path, "r") as json_file:
            gamma_line_3MeV_M2 = json.load(json_file)
 
        with open(gamma_line_M2_cats, "r") as json_file:
            gamma_line_M2_cats = json.load(json_file)
        
        ratios_data["all"]["mul_surv"]["583"]=quantiles_to_sym_error(gamma_line_M1["All"]["raw"]["583_to_2614"])
        ratios_data["all"]["mul_surv"]["609"]=quantiles_to_sym_error(gamma_line_M1["All"]["raw"]["609_to_1764"])
        ratios_data["all"]["mul_surv"][str(3125+3197)]=quantiles_to_sym_error(gamma_line_3MeV_M1["All"]["raw"]["3126M1_plus_3198M1_over_2615M1"])

        ### new ratios
        ratios_data["all"]["mul_surv"]["1377"]=quantiles_to_sym_error(gamma_line_Bi["All"]["raw"]["1378_to_1764"])
        ratios_data["all"]["mul_surv"]["1729"]=quantiles_to_sym_error(gamma_line_Bi["All"]["raw"]["1730_to_1764"])
        ratios_data["all"]["mul_surv"]["2017"]=quantiles_to_sym_error(gamma_line_Bi["All"]["raw"]["2017_to_1764"])
        ratios_data["all"]["mul_surv"]["1274"]=quantiles_to_sym_error(gamma_line_Bi["All"]["raw"]["1275_to_1764"])
        ratios_data["all"]["mul_surv"]["1415"]=quantiles_to_sym_error(gamma_line_Bi["All"]["raw"]["1416_to_1764"])


        ratios_data["all"]["mul2_sum"]["2615"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["2615M2sum_over_2615M1"])
        ratios_data["all"]["mul2_sum"]["1764"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["1764M2sum_over_1764M1"])

        ratios_data["all"]["mul2_e1"]["2615"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["2615M2E1_over_2615M1"])
        ratios_data["all"]["mul2_e1"]["583"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["583M2E1_over_2615M1"])
        ratios_data["all"]["mul2_e1"]["609"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["609M2E1_over_1764M1"])
        ratios_data["all"]["mul2_e2"]["583"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["583M2E2_over_2615M1"])
        ratios_data["all"]["mul2_e2"]["609"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["609M2E2_over_1764M1"])


        ratios_data["cat_1"]["mul2_e1"]["2615"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["2615M2cat1_over_2615M2all_e1"])
        ratios_data["cat_2"]["mul2_e1"]["2615"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["2615M2cat2_over_2615M2all_e1"])
        ratios_data["cat_3"]["mul2_e1"]["2615"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["2615M2cat3_over_2615M2all_e1"])

        ratios_data["cat_1"]["mul2_e1"]["583"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["583M2cat1_over_583M2all_e1"])
        ratios_data["cat_2"]["mul2_e1"]["583"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["583M2cat2_over_583M2all_e1"])
        ratios_data["cat_3"]["mul2_e1"]["583"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["583M2cat3_over_583M2all_e1"])

        ratios_data["cat_1"]["mul2_e1"]["609"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["609M2cat1_over_609M2all_e1"])
        ratios_data["cat_2"]["mul2_e1"]["609"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["609M2cat2_over_609M2all_e1"])
        ratios_data["cat_3"]["mul2_e1"]["609"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["609M2cat3_over_609M2all_e1"])

        ratios_data["cat_1"]["mul2_e2"]["583"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["583M2cat1_over_583M2all_e2"])
        ratios_data["cat_2"]["mul2_e2"]["583"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["583M2cat2_over_583M2all_e2"])
        ratios_data["cat_3"]["mul2_e2"]["583"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["583M2cat3_over_583M2all_e2"])

        ratios_data["cat_1"]["mul2_e2"]["609"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["609M2cat1_over_609M2all_e2"])
        ratios_data["cat_2"]["mul2_e2"]["609"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["609M2cat2_over_609M2all_e2"])
        ratios_data["cat_3"]["mul2_e2"]["609"]=quantiles_to_sym_error(gamma_line_M2_cats["All"]["raw"]["609M2cat3_over_609M2all_e2"])

        ratios_data["all"]["mul2_e1"]["583"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["583M2E1_over_2615M1"])
        ratios_data["all"]["mul2_e1"]["609"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["609M2E1_over_1764M1"])
        ratios_data["all"]["mul2_e2"]["583"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["583M2E2_over_2615M1"])
        ratios_data["all"]["mul2_e2"]["609"]=quantiles_to_sym_error(gamma_line_M2["All"]["raw"]["609M2E2_over_1764M1"])

        for group in ["top","mid_top","mid","mid_bottom","bottom"]:
            data_counts[group]={"mul_surv":{}}
            for peak in ["583.0","609.0","1765.0","2615.0","1461.0","1525.0"]:
                data_counts[group]["mul_surv"][str(int(float(peak)))]=quantiles_to_sym_error(gamma_line_groups[group]["raw"][peak],1)

    

        with open(save_data_ratio, "w") as json_file:
            json.dump(ratios_data, json_file,indent=2)
            
    #### COmpute a variable 2D binning and plot the data with this
    #### -----------------------------------------------------------------

    gamma_list_E2=np.array([277,511,583,609,768,860,934,1120,1173,1333])
    gamma_list_E1 =np.array([277,511,583,609,768,934,1120,1173,1238,1333,1408,1525,2104,2615])
    binning_E2 =    variable_binning.compute_binning(gamma_energies=gamma_list_E2,low_energy=0,high_energy=1500,gamma_binning_low=10,gamma_binning_high=10,cont_binning=50)
    binning_E1 =    variable_binning.compute_binning(gamma_energies=gamma_list_E1,low_energy=0,high_energy=3000,gamma_binning_low=10,gamma_binning_high=10,cont_binning=50)

    my_cmap = copy(plt.cm.get_cmap('viridis'))
    my_cmap.set_under('w')

    if (plot_rebin_2D):
        histo_cut_1 = utils.select_region(histo,[{"var":"sum","greater":False,"val":1460}])
        histo_cut_2 = utils.select_region(histo,[{"var":"sum","greater":True,"val":1535}])

        w,x,y = histo_cut_1.to_numpy()
        w2,x,y=histo_cut_2.to_numpy()
        w+=w2
        histo_fancy_bin = utils.fast_variable_rebinning(x,y,w,binning_E2,binning_E1)
        histo_fancy_bin=utils.normalise_histo_2D(histo_fancy_bin,100)

        w, x, y = histo_fancy_bin.to_numpy()

        fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
        mesh = axes.pcolormesh(x, y, w.T, cmap=my_cmap,vmin=0.001)
        fig.colorbar(mesh)

        axes.set_ylabel("Energy 1 [keV]")
        axes.set_xlabel("Energy 2 [keV]")

        plt.show()

    #### plot the 1D histograms in the 3 categories
    #### -------------------------------------------

    if plot_data_1D:
        names=["cat_1","cat_2","cat_3"]
        bin=5
        colors=[vset.blue,vset.magenta,vset.teal]
        for spec,spec_name in zip([0,1,2],["Energy 2","Energy 1","Energy 1 + Energy 2"]):
            fig_3, axes_3 = lps.subplots(1, 1, figsize=(4,2), sharex=True, gridspec_kw = {'hspace': 0})

            for h,n,col in zip(histo_cats,names,colors):
                if (spec==0):
                    h=h.project(0)[hist.rebin(bin)]
                elif (spec==1):
                    h=h.project(1)[hist.rebin(bin)]
                else:
                    h=utils.project_sum(h)[hist.rebin(bin)]
                h.plot(ax=axes_3,color=col,**style,label=n)
            axes_3.set_yscale("log")
            axes_3.set_xlim(0,3500)
            axes_3.set_title(spec_name)
            axes_3.set_xlabel("Energy [keV]")
            axes_3.set_ylabel("counts/ {} keV".format(bin))
            legend=axes_3.legend(loc='upper right',edgecolor="black",frameon=False, facecolor='white',framealpha=1)
            plt.show()



    #### same for a,b,c,d
    #### PLot the projections for the REGIONS a,b,c,d
    #### ---------------------------------------------------------------------------------------


    if (plot_categories_1D):
        bin=5
        hist_energy_a = histo_a.project(1)[hist.rebin(bin)]
        hist_energy_b = utils.project_sum(histo_b)[hist.rebin(bin)]
        hist_energy_c = histo_c.project(1)[hist.rebin(bin)]
        hist_energy_d = histo_d.project(0)[hist.rebin(bin)]
        fig_3, axes_3 = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})
        style["histtype"]="fill"
        style["alpha"]=0.6
        hist_energy_a.plot(ax=axes_3,color=vset.magenta,**style,label="a ($E_{1}$)")
        hist_energy_b.plot(ax=axes_3,color=vset.blue,**style,label="b ($E_{1}+E_{2}$)")
        hist_energy_c.plot(ax=axes_3,color=vset.orange,**style,label="c ($E_{1}$)")
        hist_energy_d.plot(ax=axes_3,color=vset.teal,**style,label="d ($E_{2}$)")
        axes_3.set_yscale("log")
        axes_3.set_xlim(500,3000)
        axes_3.set_xlabel("Energy [keV]")
        axes_3.set_ylabel("counts/ {} keV".format(bin))
        legend=axes_3.legend(loc='upper right',edgecolor="black",frameon=False, facecolor='white',framealpha=1,fontsize=8)






    priors_file ="cfg/priors_split.json"

    with open(priors_file,"r") as file_cfg:
        priors =json.load(file_cfg)

    priors_k42_file ="cfg/priors_K42.json"

    with open(priors_k42_file,"r") as file_cfg:
        priors_k42 =json.load(file_cfg)

    m1_cats=["all"]
    m1_cats.extend(groups)

    ### same for the MC
    ## -------------------------------------------------------------------------
    ### get the list of MC files
    list_of_mc=priors["components"]
    list_of_mc.extend(priors_k42["components"])
    plt.close("all")

    if (compute_mc):
        list_of_mc_files= os.listdir(pdf_path)

        mc ={"mul2_sum":{},"mul2_e1":{},"mul2_e2":{}}
        ## save mc output into a dictonary
        mc_counts={}

        cats=["all","cat_1","cat_2","cat_3"] #"sd_0","sd_1","sd_2","sd_3","sd_4","sd_5"]
        cats.extend(groups)
        mc={}
        for comp in tqdm(list_of_mc):
            file = comp["file"]
            name=comp["name"]
            mc_counts[name]={}
            mc[name]={}


            for cat in cats:
                if (cat in m2_cats):

                    mc[name][cat]={"mul2_sum":{},"mul2_e1":{},"mul2_e2":{},"mul_surv":{}}
                    mc_counts[name][cat]={}

                    with uproot.open(pdf_path+file,object_cache=None) as f:
                        if (cat=="all"):    
                            mc[name][cat]["mul2_e2"]=f["mul2_surv_2d"]["all"].to_hist()[0:3300,0:3300].project(0)
                            mc[name][cat]["mul2_e1"]=f["mul2_surv_2d"]["all"].to_hist()[0:3300,0:3300].project(1)
                            histo_mc =f["mul2_surv_2d"][cat].to_hist()[0:3300,0:3300]

                        else:
                            mc[name][cat]["mul2_e2"]=f["mul2_surv_2d"][cat].to_hist()[0:3300,0:3300].project(0)
                            mc[name][cat]["mul2_e1"]=f["mul2_surv_2d"][cat].to_hist()[0:3300,0:3300].project(1)
                            histo_mc = f["mul2_surv_2d"][cat].to_hist()[0:3300,0:3300]

                        histo_mc_M1 =f["mul_surv"]["all"].to_hist()
                    ##### extract the 1D histos
                    ### ----------------------------------------
                        
                    #mc[name][cat]["mul2_sum"]=utils.project_sum(histo_mc)

                    
                ## groups (M1 only)
                else:
                    mc[name][cat]={"mul_surv":{}}
                    mc_counts[name][cat]={}
                    with uproot.open(pdf_path+file,object_cache=None) as f:
                                    
                        histo_mc_M1 =f["mul_surv"][cat].to_hist()


                if (cat in m1_cats):
                    mc[name][cat]["mul_surv"]=histo_mc_M1
                    mc_counts[name][cat]["mul_surv"]={}

                ## if cat is all do the projections
             
                if (cat=="all"):
                
                    if (("Po214" in comp["name"]) or ("Pb214" in comp["name"])):
                        project_gamma_line(histo_mc,gamma_list_609,500,1500,(609,3),"e2","e1","Projection of $E_2$ 609 keV",mc_counts[name],"mul2_surv_e2_609",just_value=True,name_comb="Bi")

                    elif ("Tl208" in comp['name']):
                        project_gamma_line(histo_mc,gamma_list_583,500,2700,(583,3),"e2","e1","Projection of $E_2$ 583 keV",mc_counts[name],"mul2_surv_e2_583",just_value=True,name_comb="Tl_583_E2")
                        project_gamma_line(histo_mc,gamma_list_583,0,600,(583,3),"e1","e2","Projection of $E_1$ 583 keV",mc_counts[name],"mul2_surv_e1_583",just_value=True,name_comb="Tl_583_E1")

                        project_gamma_line(histo_mc,gamma_list_2615,0,1000,(2615,5),"e1","e2","Projection of $E_1$ 2615 keV",mc_counts[name],"mul2_surv_e1_2615",just_value=True,name_comb="Tl_2615_E1")

                if (cat in m2_cats):
                    mc_counts[name][cat]["mul2_e1"]={}
                    mc_counts[name][cat]["mul2_e2"]={}
                    mc_counts[name][cat]["mul2_sum"]={}
                
                ### peaks

                if ("Pb214" in comp["name"]):
                    peaks=[609,1764,2204,1377,1543,1729,1847,2017,1274,1415]
                elif ("Tl208" in comp["name"]):
                    peaks=[583,2615,3197,3125]
                elif ("K42" in comp["name"]):
                    peaks=[1525]
                elif ("K40" in comp["name"]):
                    peaks=[1461]
                else:
                    peaks=[]

                #### get the counts for different peaks
                ### -----------------------------------
                for peak in peaks:
                    for spec in ["mul_surv","mul2_e2","mul2_e1"]:
                        
                        if (spec=="mul_surv" and cat not in m1_cats):
                            continue
                        if (("mul2" in spec) and (cat not in m2_cats)):
                            continue
                        if peak in [1274,1415]:
                            pdf_tmp = mc[name][cat][spec][(peak-6)*1j:(peak+6)*1j][hist.rebin(4)]
                        else:
                            pdf_tmp = mc[name][cat][spec][(peak-15)*1j:(peak+15)*1j][hist.rebin(10)]

                        center= pdf_tmp[1]
                        bkg = (pdf_tmp[2]+pdf_tmp[0])/2.
                        value = center-bkg
                        if (value<0):
                            value=0
                        mc_counts[name][cat][spec][str(peak)]=value
                
                ## plot them
                if (plot_mc):
                    fig, axes = lps.subplots(1, 1, figsize=(3,2), sharex=True, gridspec_kw = {'hspace': 0})

                    histo_mc_rebin =histo_mc[hist.rebin(bin),hist.rebin(bin)]
                    w, x, y = histo_mc_rebin.to_numpy()
                    last_nonzero_y_index = np.where(w.sum(axis=0) > 0)[0][-1]
                    last_nonzero_x_index = np.where(w.sum(axis=1) > 0)[0][-1]

                    ymax = x[last_nonzero_y_index]
                    xmax= y[last_nonzero_x_index]

                    
                    norm = LogNorm(vmin=0.5, vmax=w.max())

                    mesh = axes.pcolormesh(x, y, w.T, cmap=my_cmap,norm=norm)
                    fig.colorbar(mesh)
                    axes.set_ylabel("Energy 1 [keV]")
                    axes.set_xlabel("Energy 2 [keV]")
                    if ("Ac228" in file):
                        axes.set_xlim(0,1000)
                        axes.set_ylim(0,2000)
                        high=2000
                    elif ("Pb208" in file):
                        axes.set_xlim(0,1500)
                        axes.set_ylim(0,3500) 
                        high=3500
                    elif ("K42" in file):
                        
                        axes.set_xlim(0,1500)
                        axes.set_ylim(0,3500)
                        high=3500
                    elif ("K40" in file):
                        axes.set_xlim(0,750)
                        axes.set_ylim(0,1500)
                        high=1500
                    else:
                        axes.set_xlim(0,1500)
                        axes.set_ylim(0,3000)
                        high=3000

                    plt.savefig("plots/mult_two/mc/{}_{}.png".format(file[0:-5],cat))
                    plt.close()


                    ### now the 1D histo 
                    ### ---------------------------------------------------
                    bin=10

                
                    ### rebin + make some plots
                    hist_energy_0_mc=histo_mc.project(0)[hist.rebin(bin)]
                    hist_energy_1_mc=histo_mc.project(1)[hist.rebin(bin)]
                    hist_sum_mc = utils.project_sum(histo_mc)[hist.rebin(bin)]


                    fig_2, axes_2 = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})

                    hist_sum_mc.plot(ax=axes_2,histtype="fill",alpha=0.6,color=vset.orange,**style,label="$E_{1}+E_{2}$ ")
                    hist_energy_0_mc.plot(ax=axes_2,color=vset.cyan,**style,label="$E_{2}$")
                    hist_energy_1_mc.plot(ax=axes_2,color=vset.blue,**style,label="$E_{1}$")

                    axes_2.set_yscale("linear")
                    axes_2.set_xlim(500,high)
                    axes_2.set_xlabel("Energy [keV]")
                    axes_2.set_ylabel("counts/ {} keV".format(bin))
                    legend=axes_2.legend(loc='best',edgecolor="black",frameon=False, facecolor='white',framealpha=1)

                    plt.savefig("plots/mult_two/mc/1D_{}_{}.pdf".format(file[0:-5],cat))

        bin=5

        with open(save_mc, "w") as json_file:
            json.dump(mc_counts, json_file,indent=2)


    
        bi_sum_lines=["1377","1543","1729","1847","2017","1274","1415"]


        ## Now extract ratios
        ## ------------------------------------------------------------------------------
        ratios =deepcopy(mc_counts)
        for comp in mc_counts:
            for cat in mc_counts[comp]:
                for spec in mc_counts[comp][cat]:
                    for peak in mc_counts[comp][cat][spec]:
                        if ("Pb214" in comp or "Po214" in comp):
                            norm = mc_counts[comp]["all"]["mul_surv"]["1764"]
                            tot=0
                            ## get the sum line
                            for bi in bi_sum_lines:
                                tot+=mc_counts[comp]["all"]["mul_surv"][bi]

                            ratios[comp]["all"]["mul_surv"]["bi_sum"]=100*(tot)/norm
                        elif ("Tl208" in comp):
                            norm = mc_counts[comp]["all"]["mul_surv"]["2615"]
                            if ((cat in m1_cats) and (spec=="mul_surv")):
                                ratios[comp]["all"]["mul_surv"][str(3197+3125)]=100*(mc_counts[comp]["all"]["mul_surv"]["3197"]+
                                                                            mc_counts[comp]["all"]["mul_surv"]["3125"])/norm
                        else:
                            continue
                        ratios[comp][cat][spec][peak]*=100/norm

        with open(save_mc_ratio, "w") as json_file:
            json.dump(ratios, json_file,indent=2)

    ### read from a file
    else:

        with open(save_mc, "r") as json_file:
            mc_counts = json.load(json_file)

        with open(save_mc_ratio, "r") as json_file:
            ratios = json.load(json_file)

                    


    #### now loop over ratios

    orders=["pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud","calibration_tubes","fiber_copper_inner","fiber_copper_outer","sipm","birds_nest","wls_reflector"]
    #orders=["fiber_inner","fiber_shroud","fiber_outer","fiber_copper_inner","fiber_copper","fiber_copper_outer"]
    if (get_from_data==False):
        extra="_gla_fibers"
    else:
        extra=""

    make_all_plots=True
    if make_all_plots==True:
        for cat in ratios_data:
            for spec in ratios_data[cat]:
                for peak in ratios_data[cat][spec]:
                    data_val=ratios_data[cat][spec][peak]

                    ## get mc
                    ratio_tmp={}
                
                    for comp in list_of_mc:

                        if (spec in ratios[comp["name"]][cat]) and (peak in ratios[comp["name"]][cat][spec]):
                            ratio_tmp[comp["name"][11:]]=ratios[comp["name"]][cat][spec][peak]
                            if (("Po214") in comp["name"]) or ("Pb214" in comp["name"]):
                                denom="N(1764, M1)"
                            else:
                                denom ="N(2615, M1)"

                        if (spec=="mul_surv"):
                            spec_name="M1"
                        elif (spec=="mul2_e1"):
                            spec_name="M2-E1"
                        elif (spec=="mul2_e2"):
                            spec_name="M2-E2"
                        elif (spec=="mul2_sum"):
                            spec_name="M2-Sum"
                        elif "mul2_surv_e2" in spec:
                            spec_name=" "+spec.split("e2_")[1]+" M2"
                        else:
                            spec_name=spec
                    
                    if (peak==str(3197+3125)):
                        
                        utils.plot_relative_intensities(ratio_tmp,data_val[0],data_val[1],orders,
                                        r'$\frac{{N({}, {})+N({}, {})}}{{{}}}$'.format(3125, spec_name,3197,spec_name, denom),"plots/mult_two/ratio_{}_{}_{}_{}{}.pdf".format(3125,3197,spec,cat,extra))
                    elif (peak==str("bi_sum")):
                        utils.plot_relative_intensities(ratio_tmp,data_val[0],data_val[1],orders,
                                        r'$\frac{{N(1729+ \ldots +{})}}{{{}}}$'.format(spec_name, denom),"plots/mult_two/ratio_{}_{}{}.pdf".format(spec,cat,extra))
                    elif (peak==str("Bi")):
                        utils.plot_relative_intensities(ratio_tmp,data_val[0],data_val[1],orders,
                                        r'$\frac{{N(\text{{Bi double coinc.}}, M2 )}}{{{}}}$'.format(denom),"plots/mult_two/ratio_Bi_{}_{}{}.pdf".format(spec,cat,extra))
                    elif (peak==str("Tl_2615_E1")):
                        utils.plot_relative_intensities(ratio_tmp,data_val[0],data_val[1],orders,
                                        r'$\frac{{N(\text{{Tl double coinc.}}, M2 )}}{{{}}}$'.format(denom),"plots/mult_two/ratio_Tl_{}_{}{}.pdf".format(spec,cat,extra))

                    else:
                
                        if (ratio_tmp=={}):
                            continue
                        utils.plot_relative_intensities(ratio_tmp,data_val[0],data_val[1],orders,
                                        r'$\frac{{N({}, {})}}{{{}}}$'.format(peak, spec_name, denom),"plots/mult_two/ratio_{}_{}_{}{}.pdf".format(peak,spec,cat,extra))

                    plt.close()

    #### ratios of categorie
    plot_categories=False        

    print(json.dumps(ratios_data,indent=1))

    if (plot_categories):
        cata="cat_1"
        catb="all"
        #orders=["pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud","fiber_shroud","fiber_copper"]
        #orders=["pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud","calibration_tubes","fiber_inner","fiber_shroud","fiber_outer","fiber_copper_inner","fiber_copper","fiber_copper_outer"]

        for cat,name in zip(["cat_1","cat_2","cat_3"],["same-str-adj","same-str-not-adj","diff-str"]):
            for peak in ["609","583","2615"]:
                for spec in ["mul2_e1","mul2_e2"]:
                    if (not peak in ratios_data[cat][spec]):
                        continue
                  

                    data_val = ratios_data[cat][spec][peak]

                    ratio_tmp={}
                
                    for comp in list_of_mc:

                        if (spec in ratios[comp["name"]][cat]) and (peak in ratios[comp["name"]][cat][spec]):
                            a=(mc_counts[comp["name"]][cat][spec][peak],np.sqrt(mc_counts[comp["name"]][cat][spec][peak]))
                            b=(mc_counts[comp["name"]]["all"][spec][peak],np.sqrt(mc_counts[comp["name"]]["all"][spec][peak]))
                            if (b[0]==0):
                                continue
                            ratio_tmp[comp["name"][11:]]=get_ratio(a,b)
                        if (spec=="mul_surv"):
                            spec_name="M1"
                        elif (spec=="mul2_e1"):
                            spec_name="M2-E1"
                        elif (spec=="mul2_e2"):
                            spec_name="M2-E2"
                        elif (spec=="mul2_sum"):
                            spec_name="M2-Sum"
                        elif "mul2_surv_e2" in spec:
                            spec_name=" "+spec.split("e2_")[1]+" M2"
                        else:
                            spec_name=spec
                    
                        denom = f"N({peak}, {spec_name}, all)"

                    if (ratio_tmp=={}):
                        continue
                    utils.plot_relative_intensities(ratio_tmp,data_val[0],data_val[1],orders,r'$\frac{{N({}, {}, \text{{{}}})}}{{{}}}$'.format(peak, spec_name,name, denom),"plots/mult_two/ratio_{}_{}_{}_to_all{}.pdf".format(peak,spec,cat,extra))
                    
                    #plt.show()
        plt.close("all")
        cata="cat_1"
        catb="all"
        #orders=["pen_plates","front_end_electronics","hpge_insulators","hpge_support_copper","cables","mini_shroud","calibration_tubes","fiber_inner","fiber_shroud","fiber_outer","fiber_copper_inner","fiber_copper","fiber_copper_outer"]
        cat="all"

        
        for peak,dc,dc_spec,spec in zip(["609","583","2615"],["Bi","Tl_583_E2","Tl_2615_E1"],["mul2_surv_e2_609","mul2_surv_e2_583","mul2_surv_e1_2615"],
                                        ["mul2_e2","mul2_e2","mul2_e1"]
                                        ):
           
                
            if (not peak in data_counts["all"][spec]):
                continue
            data_val=data_counts["all"][dc_spec][dc]
            data_all = data_counts["all"][spec][peak]

            data_val = get_ratio(data_val,data_all)

            ratio_tmp={}
        
            for comp in list_of_mc:

                if (spec in ratios[comp["name"]][cat]) and (peak in ratios[comp["name"]][cat][spec]):
                    a=(mc_counts[comp["name"]][cat][dc_spec][dc],np.sqrt(mc_counts[comp["name"]][cat][dc_spec][dc]))
                    b=(mc_counts[comp["name"]]["all"][spec][peak],np.sqrt(mc_counts[comp["name"]]["all"][spec][peak]))
                    if (b[0]==0):
                        continue
                    ratio_tmp[comp["name"][11:]]=get_ratio(a,b)
                if (spec=="mul_surv"):
                    spec_name="M1"
                elif (spec=="mul2_e1"):
                    spec_name="M2-E1"
                elif (spec=="mul2_e2"):
                    spec_name="M2-E2"
                elif (spec=="mul2_sum"):
                    spec_name="M2-Sum"
                elif "mul2_surv_e2" in spec:
                    spec_name=" "+spec.split("e2_")[1]+" M2"
                else:
                    spec_name=spec
            
                denom = f"N({peak}, {spec_name}, all)"

            if (ratio_tmp=={}):
                continue
            utils.plot_relative_intensities(ratio_tmp,data_val[0],data_val[1],orders,r'$\frac{{N({}, {}, \text{{{}}})}}{{{}}}$'.format(peak, spec_name,"double coinc.", denom),"plots/mult_two/ratio_{}_{}_{}_to_all{}.pdf".format(peak,spec,"DC",extra))
            
            plt.show()
    
    ### now make the relevant triptych
    cats_tmps=["all","all","all"]
    specs_tmps=["mul_surv","mul2_e1","mul2_e2"]
    peaks_tmp=["583","583","583"]
    ratios_tt=[]
    data_tt=[]
    name_tt=[]
    for c,s,p in zip(cats_tmps,specs_tmps,peaks_tmp):
        rt,dt,st,de_t =get_mc_data(priors,ratios_data,ratios,c,s,p)

        ratios_tt.append(rt)
        data_tt.append(dt)
        name_tt.append(r'$\frac{{N({}, {})}}{{{}}}$'.format(p, st, de_t))
                    
    utils.plot_relative_intensities_triptych(ratios_tt,
                                            [data_tt[0][0],data_tt[1][0],data_tt[2][0]],
                                            [data_tt[0][1],data_tt[1][1],data_tt[2][1]],orders,
                                                        name_tt,'plots/mult_two/583_tryptich.pdf')


    cats_tmps=["all","all","all"]
    specs_tmps=["mul_surv","mul2_e1","mul2_e2"]
    peaks_tmp=["609","609","609"]
    ratios_tt=[]
    data_tt=[]
    name_tt=[]
    for c,s,p in zip(cats_tmps,specs_tmps,peaks_tmp):
        rt,dt,st,de_t =get_mc_data(priors,ratios_data,ratios,c,s,p)

        ratios_tt.append(rt)
        data_tt.append(dt)
        name_tt.append(r'$\frac{{N({}, {})}}{{{}}}$'.format(p, st, de_t))
                    
    utils.plot_relative_intensities_triptych(ratios_tt,
                                            [data_tt[0][0],data_tt[1][0],data_tt[2][0]],
                                            [data_tt[0][1],data_tt[1][1],data_tt[2][1]],orders,
                                                        name_tt,'plots/mult_two/609_tryptich.pdf')
    plt.show()
                




    ### groupings plots


    ### make a histogram

    style = {

        "yerr": False,
        "flow": None,
        "lw": 0.7,
    }
    my_colors=[vset.blue,vset.teal,mset.purple,mset.olive,vset.magenta,mset.green,"orange"]
    groups_floor,_,_ = utils.get_det_types("floor",level_groups="cfg/level_groups_Sofia.json")
    groups_type,_,_ = utils.get_det_types("types",level_groups="cfg/level_groups_Sofia.json")


    for gr in [groups_floor]:
        for peak in ["583","2615","1525"]:
            if (peak=="1525"):
                iso="K42"
                orders_tmp=["cables","above","inside","outside","surface"]
                comps_use=orders_tmp
            else:
                orders_tmp=orders
                iso="Bi212Tl208"
                comps_use=["cables","birds_nest","sipm","fiber_shroud","fiber_inner","fiber_outer","fiber_copper","fiber_copper_inner","fiber_copper_outer","pen_plates","wls_reflector"]

            len_k = len(gr.keys())
            hist_z = ( Hist.new.Reg(len_k,0,len_k).Double())

            groups=list(gr.keys())
            data_tot=0
            for i in range(len_k):
                hist_z[i]=data_counts[groups[i]]["mul_surv"][peak][0]/gr[groups[i]]["exposure"]

                data_tot+=data_counts[groups[i]]["mul_surv"][peak][0]/gr[groups[i]]["exposure"]


            fig, axes = lps.subplots(1, 1, figsize=(5,3), sharex=True)
            axes.set_xticks(np.arange(len_k)+0.5, list(gr.keys()),rotation=90,fontsize=10)

            maxi = 4*max(hist_z.values())
            axes.set_ylim(0.001,maxi)
            #hist_z.plot(ax=axes,**style,color=vset.red)
        
            for i in range(len_k):
                error =data_counts[groups[i]]["mul_surv"][peak][1]/gr[groups[i]]["exposure"]
                
                ymin = (hist_z[i]-error)/maxi

                ymax = (hist_z[i]+error)/maxi
                if (i==0):
                    axes.axvspan(xmin=i,xmax=i+1,alpha=0.3,ymin=ymin,ymax=ymax,color=vset.orange,label="Data")
                else:
                    axes.axvspan(xmin=i,xmax=i+1,alpha=0.3,ymin=ymin,ymax=ymax,color=vset.orange)



            ### loop over the mc
            c=0
            for comp in orders_tmp:
                if (comp not in comps_use):
                    continue
                hist_z_tmp= ( Hist.new.Reg(len_k,0,len_k).Double())
                mc_tot=0
                for i in range(len_k):
                    hist_z_tmp[i]=mc_counts[iso+"_"+comp][groups[i]]["mul_surv"][peak]/gr[groups[i]]["exposure"]
                    mc_tot+=mc_counts[iso+"_"+comp][groups[i]]["mul_surv"][peak]/gr[groups[i]]["exposure"]

                for i in range(len_k):
                    hist_z_tmp[i]*=data_tot/mc_tot

                hist_z_tmp.plot(ax=axes,**style,color=my_colors[c],label=comp)
                c+=1

            axes.set_xlim(0,len_k)
            axes.set_ylabel("cts/kg-yr")
            axes.set_xlabel(" ")
            plt.legend(loc="best",fontsize=10,ncol=2)
            plt.show()


    


if __name__ == "__main__":
    main()
