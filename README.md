## BAT python tools
### A set of python tools to interpret the output of BAT background model fits
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk)

The tools can be divided into three groups.

#### Plotting fit outout
Several scripts have been created for the preparation of the fit.

- `manipulate_pdfs.py`: Creates a new set of PDFs for the fit 
- `variable_binning.py`: Computes a variable binning scheme for use in the fit

Some others produce plots of the outputs:

- `plot-reconstruction.py` : Plots the fit reconstruction itself
- `plot-activities.py` :  Plots the fit parameters
- `plot-correlation.py`: Plots the correlation matrix and correlation plots for parameters
- `plot-projections.py`: Projects the model onto different space
- `filter_mcmc.c`: Reduces the size of the output so that python can read it.

We created a script "examples.sh" which enables you to produce a standard set of example plots.

All have some help to explain the arguments to control the program. In general the scripts are run just using the path to the cfg file used to run the script.

`filter_mcmc.c` is the only ROOT / c++ code, it should be compiled with:

`g++ -std=c++0x filter_mcmc.c -o filter_mcmc `root-config  --cflags --glibs``
and run according to the instructions.


##### Fit preparation
Several scripts have been created for the preparation of the fit.

- `manipulate_pdfs.py`: Creates a new set of PDFs for the fit 
- `variable_binning.py`: Computes a variable binning scheme for use in the fit

##### Other scripts
Finally some scripts perform a similar (and related) analysis of gamma line ratios.
- `get-ratios.py`
- `lar-survival-prob.py`

Most of the methods used for all the steps are contained in the python module `utils`.

### Outputs/ Inputs
Configuration files (JSON) for the code is stored in '/cfg/' with plots in `/plots/`