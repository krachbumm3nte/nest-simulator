import nest
from copy import deepcopy
from pprint import pprint

# Simulation parameters
resolution = 0.1
nest.resolution = resolution
nest.set_verbosity("M_ERROR")
nest.SetKernelStatus({"local_num_threads": 3, "use_wfr": False})
nest.rng_seed = 15

init_self_pred = True
self_predicting_fb = False
self_predicting_ff = False
plasticity_fb = True
plasticity_ff = True


SIM_TIME = 100
n_runs = 10000

# Network parameters
# dims = [4, 4, 4]
dims = [15, 10, 5]
noise = True
noise_std = 0.3
target_amp = 10
stim_amp = 1
nudging = True

# Neuron parameters
g_lk_dnd = 1
g_lk_som = 1.9 # As dictated by the Mathematica solution, actual leakage is g_lk + g_D + g_A
tau_input = 1/3 # time constant for low-pass filtering the current injected into input neurons. see tests/test_curretn_injection_filter.py

lam = 0.8

g_a = 0.8
g_b_int = 1
g_b_pyr = 1
g_som = 0.8



comp_defaults = {
        'V_m': 0.0,
        'E_L': 0.0,
        'g_L': g_lk_dnd,
        'g': g_som
    }

pyr_params = {
    'soma': deepcopy(comp_defaults),
    'basal': deepcopy(comp_defaults),
    # 'apical_td': deepcopy(comp_defaults),
    'apical_lat': deepcopy(comp_defaults),
    # parameters of rate function
    'tau_m': resolution,
    'C_m': 1.0,
    'lambda': lam,
    'phi_max': 1,
    'gamma': 1,
    'beta': 1,
    'theta': 0,
}

pyr_params['basal']['g'] = g_b_pyr
# pyr_params['apical_td']['g'] = 0.0
pyr_params['apical_lat']['g'] = g_a
pyr_params['soma']['g_L'] = g_lk_som

intn_params = deepcopy(pyr_params)
intn_params['apical_lat']['g'] = 0.0
intn_params['basal']['g'] = g_b_int

# synapse parameters
# wr_pi = nest.Create('weight_recorder')
# nest.CopyModel('pyr_synapse', 'record_syn_pi', {"weight_recorder": wr_pi})

# wr_ip = nest.Create('weight_recorder')
# nest.CopyModel('pyr_synapse', 'record_syn_ip', {"weight_recorder": wr_ip})

eta_pyr_int = 0.0035
eta_int_pyr = 0.003
wmin_init, wmax_init = -1, 1
wmin, wmax = -1, 1
syn_params = {
    'synapse_model': 'pyr_synapse_rate',
    'tau_Delta': 30,
    'Wmin': wmin,
    'Wmax': wmax,
    'eta': 0.0,
    'delay': resolution,
}

# neuron parameters
pyr_model = 'pp_cond_exp_mc_pyr'
pyr_comps = nest.GetDefaults(pyr_model)["receptor_types"]

intn_model = 'pp_cond_exp_mc_pyr'
intn_comps = nest.GetDefaults(intn_model)["receptor_types"]

syn_ff_pyr_pyr = deepcopy(syn_params)
syn_ff_pyr_pyr['receptor_type'] = pyr_comps['basal']

syn_fb_pyr_pyr = deepcopy(syn_params)
syn_fb_pyr_pyr['receptor_type'] = pyr_comps['apical_lat']

syn_laminar_pyr_intn = deepcopy(syn_params)
syn_laminar_pyr_intn['receptor_type'] = intn_comps['basal']
# syn_laminar_pyr_intn['synapse_model'] = "record_syn_pi"

syn_laminar_intn_pyr = deepcopy(syn_params)
syn_laminar_intn_pyr['receptor_type'] = pyr_comps['apical_lat']
# syn_laminar_intn_pyr['synapse_model'] = "record_syn_ip"

if plasticity_fb:
    syn_laminar_intn_pyr['eta'] = eta_int_pyr
if plasticity_ff:
    syn_laminar_pyr_intn['eta'] = eta_pyr_int

# set weights after the fact because deepcopy does not enjoy copying functions.
# all_syns = [syn_ff_pyr_pyr, syn_fb_pyr_pyr, syn_laminar_intn_pyr, syn_laminar_pyr_intn]
# for s in all_syns:
#     s['weight'] = nest.random.uniform(-0.5, 0.5)
