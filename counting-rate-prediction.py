


from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
from scipy.stats import poisson
from collections import OrderedDict
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
from hist import Hist
import fc
import hist
import os
import sys
import argparse
import re
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
 


Bi=True
if (Bi==True):
    type_decay="$^{214}$Bi "
    N=171
else:
    type_decay="$^{208}$Tl "
    N=297
    
Nexp = 100
frac =0.5

Nexp*=0.366

exp = 0.366*365

eN = np.sqrt(N)

r=N/exp
eR = eN/exp


times=[]
sigs={}
for frac in [0.3,0.5,0.7]:
    sig=[]
    times=[]
    for time in np.arange(1,100):

        Ntmp = r*time*frac

        #eNtmp_plus = fc.get_upper_limit(Ntmp,0,0.683)-Ntmp
        #eNtmp_minus = Ntmp-fc.get_lower_limit(Ntmp,0,0.683)
        eNtmp_plus=np.sqrt(Ntmp)
        eNtmp_minus=np.sqrt(Ntmp)

        rtmp = r*frac
        ertmp_plus = eNtmp_minus/time
        ertmp_minus = eNtmp_plus/time

        ertmp =(ertmp_minus+ertmp_plus)/2

        sigt = (r-rtmp)/np.sqrt(ertmp*ertmp+eR*eR)

        times.append(time)
        sig.append(sigt)
    sigs[frac]=sig
    
        
fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
i=0
for key,sig in sigs.items():
   
    axes.plot(times,sig,label=str(int(100*(1-key)))+"% bkg reduction")
   
    i+=1
axes.set_xlabel("Time [days]")
axes.set_ylabel("Expected Significance [$\sigma$]")
axes.set_xlim(0,100)
axes.set_ylim(0,20)
axes.set_title(type_decay+" contamination")
plt.legend()

#### plot some examples

results={"10":[],"20":[],"30":[],"40":[],"50":[],"70":[]}
times=[]
for time in np.arange(0.1,100):
    
    for frac_str in results.keys():
        frac = float(frac_str)/100
        Ntmp = r*time*frac
        
        Ntmp=Ntmp
        eNtmp_plus = np.sqrt(Ntmp) #fc.get_upper_limit(Ntmp,0,0.683)-Ntmp
        eNtmp_minus = np.sqrt(Ntmp) #Ntmp-fc.get_lower_limit(Ntmp,0,0.683)
        
  
        rtmp = r*frac
        ertmp_plus = eNtmp_minus/time
        ertmp_minus = eNtmp_plus/time

        rtmp = 100*(Ntmp/time)/r
        ertmp_plus = 100*(eNtmp_plus/time)/r
        ertmp_minus = 100*(eNtmp_minus/time)/r

        results[frac_str].append((rtmp,ertmp_minus,ertmp_plus))


    times.append(time)

for frac_str in results.keys():
    fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
    med =[]
    low=[]
    high=[]
    for val in results[frac_str]:
        med.append(val[0])
        low.append(val[0]-val[1])
        high.append(val[0]+val[2])
    frac  =float(frac_str)

    axes.fill_between(times,np.array(low),np.array(high), color=vset.cyan, alpha=0.4,label="Without OB")
    axes.plot(times,med,color=vset.blue)
    axes.fill_between(times,np.ones(len(times))*(100-100*eR/r),np.ones(len(times))*(100+100*eR/r), color=vset.orange, alpha=0.4,label="With IB")
    axes.plot(times,np.ones(len(times))*(100),color=vset.red)
    axes.legend(loc="upper right")
    axes.set_xlabel("Time [days]")
    frac_red = 100-100*frac
    axes.set_title(f" {100-frac} % bkg reduction ({type_decay})")
    axes.set_ylabel("Normalised gamma counting rate [%]")
    axes.set_xlim(1,50)
    axes.set_ylim(0,150)
    plt.show()


fracs=[]
times=[]
for time in np.arange(1,100):

    sigt=0
    frac=1
    while (sigt<3):
        frac-=0.001
        Ntmp = r*time*frac
        eNtmp = np.sqrt(Ntmp)

        rtmp = r*frac
        
        ertmp = eNtmp/time
            

        sigt = (r-rtmp)/np.sqrt(ertmp*ertmp+eR*eR)
    times.append(time)
    fracs.append(frac)
fracs=np.array(fracs)
fig, axes2 = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
axes2.set_xlabel("Time [days]")
axes2.set_ylabel("")
axes2.set_xlabel("Time [days]")
axes2.set_ylabel("Fraction bkg reduction sensiitivty")
axes2.plot(times,(100-100*fracs),color=vset.blue)
axes2.set_title(type_decay)
plt.show()
