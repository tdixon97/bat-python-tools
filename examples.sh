#/bin/bash


python plot-taup.py -d by_type -D icpc,bege,ppc,coax,cat_a_e1,cat_b_sum,cat_c_e1,cat_d_e2  -N "M1 (ICPC), M1  (BeGe),M1 (PPC), M1 (COAX), M2 a,M2 b, M2 c,M2 d" -l 500,500,500,500,400,1400,1500,500 -u 4000,4000,4000,4000,1400,1600,3000,1500 -s linear -w 4 -o ../hmixfit/results/hmixfit-l200a_vancouver_workshop_v0_2_new_m2/histograms.root -f l200a_vancouver23_dataset_v0_3_split_geometry
