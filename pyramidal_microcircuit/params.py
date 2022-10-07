import nest
from copy import deepcopy
from pprint import pprint

resolution = 0.1
nest.resolution = resolution
nest.set_verbosity("M_ERROR")
nest.SetKernelStatus({"local_num_threads": 1})
nest.rng_seed = 1

"""intn_params = {
    't_ref': 3.0,
    'g_sp': 0.8,
    'g_ps': 1.0,
    'soma': {
        'V_m': 0.0,
        'C_m': 1.0,
        'E_L': 0.0,
        'g_L': 0.1,
        'E_ex': 14.0 / 3.0,
        'E_in': -1.0 / 3.0,
        'tau_syn_ex': 3.0,
        'tau_syn_in': 3.0,
    },
    'dendritic': {
        'V_m': 0.0,
        'C_m': 1.0,
        'E_L': 0.0,
        'g_L': 0.1,
        'tau_syn_ex': 3.0,
        'tau_syn_in': 3.0,
    },
    # parameters of rate function
    'phi_max': 0.15,
    'rate_slope': 0.5,
    'beta': 5.0,
    'theta': 1.0,
}
"""
comp_defaults = {
        'V_m': 0.0,
        'E_L': 0.0,
        'g_L': 0.08,
        'g': 0.8
    }

pyr_params = {
    't_ref': 3.0,
    'soma': deepcopy(comp_defaults),
    'basal': deepcopy(comp_defaults),
    'apical_td': deepcopy(comp_defaults),
    'apical_lat': deepcopy(comp_defaults),
    # parameters of rate function
    'C_m': 1.0,
    'lambda': 0.5

}
pyr_params['basal']['g'] = 1.0
pyr_params['soma']['g_L'] = 0.01

intn_params = deepcopy(pyr_params)

intn_params['apical_td']['g'] = 0.0
intn_params['apical_lat']['g'] = 0.0

# synapse params:
wr = nest.Create('weight_recorder')
nest.CopyModel('pyr_synapse', 'record_syn', {"weight_recorder": wr})

syn_params = {
    'synapse_model': 'record_syn',
    'tau_Delta': 30,
    'Wmin': -10.0,
    'Wmax': 10.0,  # TODO: verify
    'eta': 0.0,
    'delay': resolution,
}

static_syn_params = {
    'delay': resolution,
}

# neuron parameters
pyr_model = 'pp_cond_exp_mc_pyr'
pyr_comps = nest.GetDefaults(pyr_model)["receptor_types"]
print(pyr_comps)
intn_model = 'pp_cond_exp_mc_pyr'
intn_comps = nest.GetDefaults(intn_model)["receptor_types"]

syn_ff_pyr_pyr = deepcopy(static_syn_params)
syn_ff_pyr_pyr['receptor_type'] = pyr_comps['basal']

syn_fb_pyr_pyr = deepcopy(static_syn_params)
syn_fb_pyr_pyr['receptor_type'] = pyr_comps['apical_td']

syn_laminar_pyr_intn = deepcopy(syn_params)
syn_laminar_pyr_intn['receptor_type'] = intn_comps['basal']
# syn_laminar_pyr_intn['eta'] = 0.02375
syn_laminar_pyr_intn['eta'] = 0.0001

syn_laminar_intn_pyr = deepcopy(syn_params)
syn_laminar_intn_pyr['receptor_type'] = pyr_comps['apical_td']
syn_laminar_intn_pyr['eta'] = 0.0001
# syn_laminar_intn_pyr['eta'] = 0.0005

print(pyr_comps)
print(syn_laminar_intn_pyr)

# set weights after the fact because deepcopy does not enjoy copying functions.
all_syns = [syn_ff_pyr_pyr, syn_fb_pyr_pyr, syn_laminar_intn_pyr, syn_laminar_pyr_intn]
for s in all_syns:
    s['weight'] = nest.random.uniform(-1, 1)
