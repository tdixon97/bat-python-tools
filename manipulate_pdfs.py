"""
manipulate_pdfs.py
author: Toby Dixon (toby.dixon.23@ucl.ac.uk)
Make a manipulation of the pdfs for the fit (data and mc are treated symmetrically) to produce new fit pdfs
"""

from __future__ import annotations

import json
import logging
import os

import uproot
import utils
from hist import Hist

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_list_channels(file):
    """Get the list of directories inside a root file"""

    chan_keys = [key[9:].split(";")[0] for key in file.keys() if "mul_surv/ch" in key]
    return chan_keys


### file paths
is_LEGEND = True


#### LEGEND
if is_LEGEND:
    pdf_path = "/home/tdixon/LEGEND/BackgroundModel/hmixfit/inputs/pdfs/l200a-pdfs_neutrino24_v0.0/l200a-pdfs/"
    data_path = "/home/tdixon/LEGEND/BackgroundModel/hmixfit/inputs/data/datasets/l200a-neutrino24_v0.0.root"
    extra_label = "0"
    list_of_pdfs = os.listdir(pdf_path)
    pdf_out_path = pdf_path[0:-1] + "." + extra_label + "/"
    os.makedirs(pdf_out_path, exist_ok=True)
    data_out_path = data_path[0:-5] + "." + extra_label + ".root"
    m2_name = "mul2_surv_2d"
    m2_cats = ["all", "cat_1", "cat_2", "cat_3", "sd_0", "sd_1", "sd_2", "sd_3", "sd_4", "sd_5"]

    m2_e1_out_name = "mul2_surv_e1"
    m2_e2_out_name = "mul2_surv_e2"
    is_sort_m2 = False
    is_sum = False
    input_list = [data_path]
    output_list = [data_out_path]
    char_2_remove = 2

    logger.info(f"Running with: PDF path {pdf_path}")
    logger.info(f"              Data path {data_path}")
    logger.info(f"              Extra label {extra_label}")

else:
    is_sum = False
    data_path_pII = "/home/tdixon/LEGEND/BackgroundModel/hmixfit/inputs/data/datasets/gerda-data-bkgmodel-phaseII-v07.01-orig.root"
    data_path_pIIp = "/home/tdixon/LEGEND/BackgroundModel/hmixfit/inputs/data/datasets/gerda-data-bkgmodel-phaseIIplus-v07.01-orig.root"
    data_path_out = "/home/tdixon/LEGEND/BackgroundModel/hmixfit/inputs/data/datasets/gerda-data-bkgmodel-phaseIImerge-v07.01-orig.root"

    extra_label = "_two_dim_m2_cut"

    data_out_path_1 = data_path_pII[0:-5] + "_" + extra_label + ".root"
    data_out_path_2 = data_path_pIIp[0:-5] + "_" + extra_label + ".root"

    m2_name = "raw/M2_enrE1vsE2"
    m2_e1_out_name = "raw/M2_enrE1"
    m2_e2_out_name = "raw/M2_enrE2"
    is_sort_m2 = True
    input_list = [data_path_pII, data_path_pIIp]
    output_list = [data_out_path_1, data_out_path_2]
    char_2_remove = 3

### m2 regions
m2_regions_path = "cfg/m2_regions.json"
with open(m2_regions_path) as file:
    m2_regions = json.load(file)


## create a list of input and output files

if is_LEGEND:
    for pdf in list_of_pdfs:
        input_list.append(pdf_path + pdf)
        output_list.append(pdf_out_path + pdf)

### which manipulations to do
is_pdf_per_string = False
is_pdf_per_vertical_group = False
is_m2_projections = True
is_m2_group_projection = True
is_peak_analysis_pdfs = True
peak_analysis = {
    "k_peaks": {"var": "mul_surv", "bins": [1440, 1456, 1467, 1520, 1530, 1545]},
    "tl_2615": {"var": "mul_surv", "bins": [2600, 2610, 2630, 2650]},
    "tl_583_bi_609": {"var": "mul_surv", "bins": [570, 580, 586, 606, 612, 622]},
    "bi_1764": {"var": "mul_surv", "bins": [1749, 1759, 1769, 1779]},
}

is_2D_M2 = False
is_first = False

logger.info(f"Produce string pdfs    : {is_pdf_per_string}")
logger.info(f"Produce vertical pdfs  : {is_pdf_per_vertical_group}")
logger.info(f"Produce vertical pdfs  : {is_pdf_per_vertical_group}")
logger.info(f"Produce m2 projections : {is_m2_projections}, {is_m2_group_projection}")
logger.info(f"Make rebinned 2D pdf   : {is_2D_M2}")
logger.info(f"Make peak analysis pdfs: {is_peak_analysis_pdfs}")

#### first open the data
#### -----------------------------------
for input_path, output_path in zip(input_list, output_list):
    logger.info(f">>> {input_path.split('/')[-1]} -> {output_path.split('/')[-1]}")
    with uproot.open(input_path, object_cache=None) as data_file:
        if is_first is False:
            chans = get_list_channels(file=data_file)
            is_first = False
        hists = utils.get_list_of_not_directories(file=data_file)

        ### create the output file and copy everything (to start)
        with uproot.recreate(output_path) as output_file:
            hists_old = utils.get_list_of_not_directories(file=output_file)

            ## try copying everything
            for obj in hists:
                ## dont save these histos
                if ("ID" in obj) or ("M1_all" in obj):
                    continue
                if isinstance(data_file[obj], uproot.models.TObjString.Model_TObjString):
                    ### special saving need for TObjString (seems a bug)
                    output_file[obj.split(";")[0]] = data_file[obj].tojson()["fString"]
                else:
                    ## be sure to remove the

                    output_file[obj.split(";")[0]] = data_file[obj]

            ### now the new pdfs
            ### PER STRING
            ### -----------------------------------------------------------------------
            if is_pdf_per_string == True:
                groups, _, _ = utils.get_det_types("string")
                for string in groups.keys():
                    first = True
                    hist_tmp = None
                    for i in range(len(groups[string]["names"])):
                        if ("mul_surv/" + groups[string]["names"][i] + ";1") in data_file.keys():
                            if first == False:
                                hist_tmp += data_file[
                                    "mul_surv/{}".format(groups[string]["names"][i])
                                ].to_hist()
                            else:
                                hist_tmp = data_file[
                                    "mul_surv/{}".format(groups[string]["names"][i])
                                ].to_hist()
                                first = False
                    if hist_tmp != None:
                        output_file[f"mul_surv/string_{string}"] = hist_tmp

            #### PER FLOOR
            #### -------------------------------------------------------------------

            if is_pdf_per_vertical_group == True:
                groups, _, _ = utils.get_det_types(
                    "floor", level_groups="cfg/level_groups_Sofia.json"
                )

                for floor in groups.keys():
                    first = True
                    hist_tmp = None
                    for i in range(len(groups[floor]["names"])):
                        if ("mul_surv/" + groups[floor]["names"][i] + ";1") in data_file.keys():
                            if first == False:
                                hist_tmp += data_file[
                                    "mul_surv/{}".format(groups[floor]["names"][i])
                                ].to_hist()
                            else:
                                hist_tmp = data_file[
                                    "mul_surv/{}".format(groups[floor]["names"][i])
                                ].to_hist()
                                first = False

                    if hist_tmp != None:
                        output_file[f"mul_surv/{floor}"] = hist_tmp

            ### projections of m2 groups
            ### ---------------------------------------------------------------------

            if is_m2_group_projection == True:
                if is_LEGEND == False:
                    raise NotImplementedError("Groups are only implemented for LEGEND not GERDA")

                for cat in ["all"]:
                    if cat == "all":
                        mul_spec = data_file["mul2_surv_2d/all"].to_hist()
                    else:
                        mul_spec = data_file[f"mul2_surv_2d/{cat}"].to_hist()

                    for region in m2_regions["regions"]:
                        hist_tmp = utils.select_region(histo=mul_spec, cuts=region["cuts"])
                        if region["var"] == "e1":
                            hist_1d = hist_tmp.project(1)
                        elif region["var"] == "e2":
                            hist_1d = hist_tmp.project(0)
                        else:
                            hist_1d = utils.project_sum(hist_tmp)

                        if cat == "all":
                            output_file[
                                "mul2_surv/cat_{}_{}".format(region["name"], region["var"])
                            ] = hist_1d
                        else:
                            output_file[
                                "mul2_surv/{}_cat_{}_{}".format(cat, region["name"], region["var"])
                            ] = hist_1d

            ### K analysis pdfs - slightly harder
            if is_peak_analysis_pdfs == True:
                for peak in peak_analysis:
                    bins = peak_analysis[peak]["bins"]

                    for b in range(len(bins) - 1):
                        histo = Hist.new.Reg(len(chans), 0, len(chans)).Double()

                        for idx in range(len(chans)):
                            if (
                                peak_analysis[peak]["var"] + "/" + chans[idx] + ";1"
                            ) in data_file.keys():
                                spec = data_file[
                                    peak_analysis[peak]["var"] + "/" + chans[idx]
                                ].to_hist()

                                histo[idx] = utils.integrate_hist(spec, bins[b], bins[b + 1])

                        output_file[
                            peak_analysis[peak]["var"] + "_" + peak + "/" + peak + f"_ids_{b}"
                        ] = histo

            ## save (basic) m2 projections
            if is_m2_projections == True:
                ### ie for LEGEND
                if is_sort_m2 == False:
                    for cat in m2_cats:
                        if cat == "all":
                            mul_spec = data_file[m2_name + "/all"].to_boost()

                            output_file[m2_e1_out_name + "/all"] = mul_spec.project(1)
                            output_file[m2_e2_out_name + "/all"] = mul_spec.project(0)
                        else:
                            mul_spec = data_file[m2_name + f"/{cat}"].to_boost()

                            output_file[m2_e1_out_name + f"/{cat}"] = mul_spec.project(1)
                            output_file[m2_e2_out_name + f"/{cat}"] = mul_spec.project(0)
                else:
                    ## build the 1D histo
                    mul_spec = data_file[m2_name].to_boost()

                    mul_spec_plus = utils.select_region(
                        mul_spec, [{"var": "diff", "greater": True, "val": 0}]
                    )

                    mul_spec_minus = utils.select_region(
                        mul_spec, [{"var": "diff", "greater": False, "val": 0}]
                    )
                    hist_e1 = mul_spec_plus.project(1)
                    hist_e2 = mul_spec_plus.project(0)
                    m0 = mul_spec_minus.project(0)
                    m1 = mul_spec_minus.project(1)
                    for i in range(hist_e1.size - 2):
                        hist_e1[i] += m0[i]
                        hist_e2[i] += m1[i]

                    output_file[m2_e1_out_name] = hist_e1
                    output_file[m2_e2_out_name] = hist_e2

            if is_2D_M2 == True:
                for cat in ["all"]:
                    mul_spec = data_file[m2_name + "/all"].to_boost()[200:1500, 200:3000]

                    mul_spec_less_K = utils.select_region(
                        mul_spec,
                        [
                            {"var": "sum", "greater": True, "val": 1400},
                            {"var": "sum", "greater": False, "val": 1545},
                        ],
                    )
                    w, x, y = (mul_spec - mul_spec_less_K).to_numpy()
                    w = w.astype("float32")
                    output_file["mul2_surv_2d/without_k_lines"] = (w, x, y)
                    output_file["mul2_surv_2d/just_k_lines"] = data_file[
                        "mul2_surv/all"
                    ].to_boost()[1400:1545]
