import nest
from copy import deepcopy
from pprint import pprint

resolution = 0.1

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
        'C_m': 1.0,
        'E_L': 0.0,
        'g_L': 0.1,
        'E_ex': 14.0 / 3.0,
        'E_in': -1.0 / 3.0,
        'tau_syn_ex': 3.0,
        'tau_syn_in': 3.0,
        'g': 0.8
    }

pyr_params = {
    't_ref': 3.0,
    'soma': deepcopy(comp_defaults),
    'basal': deepcopy(comp_defaults),
    'apical_td': deepcopy(comp_defaults),
    'apical_lat': deepcopy(comp_defaults),
    # parameters of rate function
    'phi_max': 0.15,
    'rate_slope': 0.5,
    'beta': 5.0,
    'theta': 1.0,
}
pyr_params['basal']['g'] = 1.0

intn_params = deepcopy(pyr_params)

intn_params['apical_td']['g'] = 0.0
intn_params['apical_lat']['g'] = 0.0

# synapse params:

syn_params = {
    'synapse_model': 'pyr_synapse',
    'tau_Delta': 30,
    'Wmin': -1.0,
    'Wmax': 1.0,  # TODO: verify
    'delay': resolution,
}

stat_syn_params = deepcopy(syn_params)
stat_syn_params['synapse_model'] = 'static_synapse'
del stat_syn_params['tau_Delta']
del stat_syn_params['Wmin']
del stat_syn_params['Wmax']


# neuron parameters
pyr_model = 'pp_cond_exp_mc_pyr'
pyr_comps = nest.GetDefaults(pyr_model)["receptor_types"]
intn_model = 'pp_cond_exp_mc_pyr'
intn_comps = nest.GetDefaults(intn_model)["receptor_types"]

syn_ff_pyr_pyr_e = deepcopy(stat_syn_params)
syn_ff_pyr_pyr_e['receptor_type'] = pyr_comps['basal_exc']

syn_ff_pyr_pyr_i = deepcopy(syn_ff_pyr_pyr_e)
syn_ff_pyr_pyr_i['receptor_type'] = pyr_comps['basal_inh']


syn_fb_pyr_pyr_e = deepcopy(stat_syn_params)
syn_fb_pyr_pyr_e['receptor_type'] = pyr_comps['apical_td_exc']


# syn_fb_pyr_pyr_i = deepcopy(syn_fb_pyr_pyr_e)
# syn_fb_pyr_pyr_i['receptor_type'] = pyr_comps['apical_td_inh']


syn_laminar_pyr_intn_e = deepcopy(syn_params)
syn_laminar_pyr_intn_e['receptor_type'] = intn_comps['basal_exc']
syn_laminar_pyr_intn_e['eta'] = 0.0002375

syn_laminar_pyr_intn_i = deepcopy(syn_laminar_pyr_intn_e)
syn_laminar_pyr_intn_i['receptor_type'] = intn_comps['basal_inh']

syn_laminar_intn_pyr_i = deepcopy(syn_params)
syn_laminar_intn_pyr_i['receptor_type'] = pyr_comps['apical_lat_inh']
syn_laminar_intn_pyr_i['eta'] = 0.0005

print(pyr_comps)
print(syn_laminar_intn_pyr_i)

# syn_laminar_intn_pyr_e = deepcopy(syn_laminar_intn_pyr_i)
# syn_laminar_intn_pyr_e['receptor_type'] = pyr_comps['apical_lat_exc']

# set weights after the fact because deepcopy does not enjoy copying functions.
all_syns = [syn_ff_pyr_pyr_e, syn_ff_pyr_pyr_i, syn_fb_pyr_pyr_e, syn_laminar_intn_pyr_i, syn_laminar_pyr_intn_e, syn_laminar_pyr_intn_i]
for s in all_syns:
    s['weight'] = nest.random.uniform(-1, 1)

