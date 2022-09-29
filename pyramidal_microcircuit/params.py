import nest
from copy import copy

resolution = 0.1

intn_params = {
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


pyr_params = {
    't_ref': 3.0,
    'g_som': 0.8,
    'g_a': 0.8,
    'g_b': 1.0,
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
    'basal': {
        'V_m': 0.0,
        'C_m': 1.0,
        'E_L': 0.0,
        'g_L': 0.1,
        'tau_syn_ex': 3.0,
        'tau_syn_in': 3.0,
    },
    'apical': {
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

# synapse params:

syn_params = {
    'synapse_model': 'pyr_synapse',
    'tau_Delta': 30,
    'weight': nest.random.uniform(0, 1),
    'Wmin': 0.0,
    'Wmax': 3.0,  # TODO: verify
    'delay': resolution,
}


# neuron parameters
pyr_model = 'pp_cond_exp_mc_pyr'
pyr_comps = nest.GetDefaults(pyr_model)["receptor_types"]
intn_model = 'pp_cond_exp_mc_urbanczik'
intn_comps = nest.GetDefaults(intn_model)["receptor_types"]

syn_ff_pyr_pyr = copy(syn_params)
syn_ff_pyr_pyr['receptor_type'] = pyr_comps['basal_exc']
syn_ff_pyr_pyr['eta'] = 0

syn_fb_pyr_pyr = copy(syn_params)
syn_fb_pyr_pyr['receptor_type'] = pyr_comps['apical_exc']
syn_fb_pyr_pyr['eta'] = 0

syn_laminar_pyr_intn = copy(syn_params)
syn_laminar_pyr_intn['receptor_type'] = intn_comps['dendritic_exc']
syn_laminar_pyr_intn['eta'] = 0.0002375
syn_laminar_pyr_intn['synapse_model'] = 'urbanczik_synapse'

syn_laminar_intn_pyr = copy(syn_params)
syn_laminar_intn_pyr['receptor_type'] = pyr_comps['apical_inh']
syn_laminar_intn_pyr['eta'] = 0.0005


print(pyr_comps)
