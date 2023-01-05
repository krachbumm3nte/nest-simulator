import nest
from copy import deepcopy
import numpy as np
import os
from datetime import datetime

# environment parameters
root = f"/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"

imgdir = os.path.join(root, "plots")
datadir = os.path.join(root, "data")
for p in [root, imgdir, datadir]:
    os.mkdir(p)

# Simulation parameters
delta_t = 0.1
sqrt_dt = np.sqrt(delta_t)
threads = 6
record_interval = 75

nest.set_verbosity("M_ERROR")
nest.resolution = delta_t
nest.SetKernelStatus({"local_num_threads": threads, "use_wfr": False})
nest.rng_seed = 15

nest.SetDefaults("multimeter", {'interval': record_interval})

nest.SetKernelStatus({"data_path": datadir})


init_self_pred = False
self_predicting_fb = False
self_predicting_ff = False
plasticity = False
bogo_plasticity = True


SIM_TIME = 100
n_runs = 10000

# Network parameters
noise = True
sigma = 0.1
noise_factor = sigma * sqrt_dt
target_amp = 10
stim_amp = 1
nudging = True

# transfer function parameters
gamma = 1
beta = 1
theta = 0
# gamma = 0.1
# beta = 1
# theta = 3


# Neuron parameters
tau_x = 3
tau_input = 1/3  # time constant for low-pass filtering the current injected into input neurons.

g_l = 0.1
g_lk_dnd = 1

g_a = 0.8
g_d = 1
g_som = 0.8

g_si = 0.8  # interneuron nudging conductance
g_s = 0.8  # output neuron nudging conductance

lambda_ah = g_a / (g_d + g_a + g_l)

comp_defaults = {
    'V_m': 0.0,
    'E_L': 0.0,
    'g_L': g_lk_dnd,
    'g': g_som
}

pyr_params = {
    'soma': deepcopy(comp_defaults),
    'basal': deepcopy(comp_defaults),
    'apical_lat': deepcopy(comp_defaults),
    # parameters of rate function
    'tau_m': 1,
    'C_m': 1.0,
    'lambda': g_si,
    'phi_max': 1,
    'gamma': gamma,
    'beta': beta,
    'theta': theta,
    'use_phi': True
}

pyr_params['basal']['g'] = g_d
# pyr_params['apical_td']['g'] = 0.0
pyr_params['apical_lat']['g'] = g_a
pyr_params['soma']['g_L'] = g_l
# misappropriation of somatic conductance. this is the effective leakage conductance now!
pyr_params['soma']['g'] = g_l + g_d + g_si

intn_params = deepcopy(pyr_params)
intn_params['apical_lat']['g'] = 0.0

# to replace the low pass filtering of the input, input neurons have both
# injected current and leakage conductance attenuated.
# Additionally, the dendritic compartments are silenced, and membrane voltage is
# transmitted to the hidden layer without the nonlinearity phi.
input_params = deepcopy(pyr_params)
input_params["soma"]["g"] = tau_input
input_params["use_phi"] = False
input_params['basal']['g'] = 0
input_params['tau_m'] = tau_input


# synapse parameters
# wr = nest.Create('weight_recorder')
# nest.CopyModel('pyr_synapse_rate', 'record_syn', {"weight_recorder": wr})

# wr_ip = nest.Create('weight_recorder')
# nest.CopyModel('pyr_synapse', 'record_syn_ip', {"weight_recorder": wr_ip})

if plasticity:
    eta_yh = 0.01
    eta_hx = eta_yh / lambda_ah
    eta_ih = 0.01 / lambda_ah
    eta_hi = 5 * eta_ih
else:
    eta_yh = 0
    eta_hx = 0
    eta_hi = 0
    eta_ih = 0
eta_hy = 0
if bogo_plasticity:
    eta_yh *= 100
    eta_hx *= 100
    eta_hi *= 100
    eta_ih *= 100


eta_ih = 0.0002
eta_hi = 0.0005

wmin_init, wmax_init = -1, 1
wmin, wmax = -2, 2
tau_delta = 30

# TODO: set up weight recorder.
syn_params = {
    'synapse_model': 'pyr_synapse_rate',
    'tau_Delta': tau_delta,
    'Wmin': wmin,
    'Wmax': wmax,
    'eta': 0.0,
    'delay': delta_t,
}

# neuron parameters
pyr_model = 'pp_cond_exp_mc_pyr'
pyr_comps = nest.GetDefaults(pyr_model)["receptor_types"]

intn_model = 'pp_cond_exp_mc_pyr'
intn_comps = nest.GetDefaults(intn_model)["receptor_types"]

basal_dendrite = pyr_comps['basal']
apical_dendrite = pyr_comps['apical_lat']

syn_hx = deepcopy(syn_params)
syn_hx['receptor_type'] = basal_dendrite

syn_yh = deepcopy(syn_params)
syn_yh['receptor_type'] = basal_dendrite

syn_hy = deepcopy(syn_params)
syn_hy['receptor_type'] = apical_dendrite

syn_ih = deepcopy(syn_params)
syn_ih['receptor_type'] = basal_dendrite

syn_hi = deepcopy(syn_params)
syn_hi['receptor_type'] = apical_dendrite

syn_hi['eta'] = eta_hi
syn_ih['eta'] = eta_ih
syn_hx['eta'] = eta_hx
syn_yh['eta'] = eta_yh

# set weights after the fact because deepcopy does not enjoy copying functions.
# all_syns = [syn_ff_pyr_pyr, syn_fb_pyr_pyr, syn_laminar_intn_pyr, syn_laminar_pyr_intn]
# for s in all_syns:
#     s['weight'] = nest.random.uniform(-0.5, 0.5)


def phi(x):
    return np.log(1 + np.exp(x))
    return gamma * np.log(1 + np.exp(beta * (x - theta)))
    return 1 / (1.0 + np.exp(-x))


def phi_inverse(x):
    return (1 / beta) * (beta * theta + np.log(np.exp(x/gamma) - 1))
