This drive containins the electronic supplementary material for the Master's thesis
"learning in cortical microcircuits with multi-compartment pyramidal neurons".
The thesis was written by Johannes Gille in May 2023.

This file will briefly outline the structure of the supplement. The base
directory contains a variant of the NEST simulator, which is published under the
GPLv2+ license (https://www.gnu.org/licenses/old-licenses/gpl-2.0.html),
as is all other code in this directory tree. The NEST code included here was
last pulled from the public repository on December 20, 2022 and therefore lies
somewhere between NEST versions 3.3 and 3.4. 

Under the `models` directory, the neuron (`pp_cond_exp_mc_pyr`,
`rate_neuron_pyr`) and synapse models (`pyr_synapse`, `pyr_synapse_rate`) are
located with corresponding .cpp and .h files. Furthermore, the
`pyr_archiving_node` responsible for storage and readout of dendritic error, is
located in the `nestkernel` directory. Other changes to the simulator largely
serve to embed these files and make them accessible from the PyNEST API. The
easiest way to compile the NEST simulator (which is required to run the model)
is through conda. The `requirements.txt` file in the base directory specifies
the required packages. After creating a new environment with 

`$ conda create -n <environment-name> --file requirements.txt` 


a guide from the NEST documentation
(https://nest-simulator.readthedocs.io/en/v3.4/installation/condaenv_install.html#condaenv) should simplify setup of the simulator. To test if setup was successful, executing the main
training script

`$ python3 scripts/train_network.py`

without arguments should launch training of a spiking dendritic error network on the Bars
dataset using default parameters


The project directory dendritic error network (roughly) follows the
“Good research code handbook“ (Mineault and Community, 2021): 

- `data` contains
all Figures in this thesis, as well as some additional ones which were not
included. 


- `results` contains all relevant simulation results. Parameter studies
have their own sub- folders, titled with the prefix par study . Each simulation
stores its full parameter set (params.json), initial weights (init
weights.json), final weights (weights.json) and training progress
(progress.json). Intermediate weights are stored every few epochs during
training in the data subdirectory. If plots were created to validate training
progress, they are found in the plots folder. 

- `experimental_configs` contains
JSON files to parameterize experiments. In these files, only the subset of
parameters needs to be specified which deviates from the default (cf.
Supplementary table S2).

- `scripts` contains scripts to be executed. The vast majority of them were written to vali-
date individual assumptions made in this thesis. Of interest to perform experiments are
train network.py and parameter study.py, which both support the --help parameter
and will therefore not be detailed here.

- `src` contains a python module of reusable code, including networks, datasets, default
parameters and some utilities. It can be added to the conda environment via $ pip
install -e . from the project’s base directory.

- `tests` contains a test-suite which was used to validate the NEST implementation and
fine-tune parameters. As it was only strictly necessary during the initial stages, it has
not been updated throughout. Therefore most tests currently fail.