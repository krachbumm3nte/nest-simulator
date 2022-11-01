import nest
from copy import deepcopy
from pprint import pprint

resolution = 0.1
nest.resolution = resolution
nest.set_verbosity("M_ERROR")
nest.SetKernelStatus({"local_num_threads": 3})
nest.rng_seed = 156


g_a = 0.8
g_b = 1.0
g_lk_dnd = 0.8
g_lk_som = 0.8
g_som = 0.8

lam = g_som / (g_lk_som + g_b + g_som)

comp_defaults = {
        'V_m': 0.0,
        'E_L': 0.0,
        'g_L': g_lk_dnd,
        'g': g_som
    }

pyr_params = {
    'soma': deepcopy(comp_defaults),
    'basal': deepcopy(comp_defaults),
    'apical_td': deepcopy(comp_defaults),
    'apical_lat': deepcopy(comp_defaults),
    # parameters of rate function
    'C_m': 1.0,
    'lambda': lam,
    # 'phi_max': 1,
    # 'rate_slope': 1,
    # 'beta': 1,
    # 'theta': 0,
    'phi_max': 0.55,
    'rate_slope': 0.5,
    'beta': 3.0,
    'theta': 0.5,
    't_ref': 0.5,

}

pyr_params['basal']['g'] = g_b
pyr_params['apical_td']['g'] = 0.0
pyr_params['apical_lat']['g'] = g_a
pyr_params['soma']['g_L'] = g_lk_som

intn_params = deepcopy(pyr_params)
intn_params['apical_lat']['g'] = 0.0

# synapse params:
wr = nest.Create('weight_recorder')
nest.CopyModel('pyr_synapse', 'record_syn', {"weight_recorder": wr})

syn_params = {
    'synapse_model': 'record_syn',
    'tau_Delta': 30,
    'Wmin': -1.0,
    'Wmax': 1.0,  # TODO: verify
    'eta': 0.0,
    'delay': resolution,
}

static_syn_params = {
    'synapse_model': 'record_syn',
    'tau_Delta': 30,
    'Wmin': -1.0,
    'Wmax': 1.0,  # TODO: verify
    'eta': 0.0,
    'delay': resolution,
}

# neuron parameters
pyr_model = 'pp_cond_exp_mc_pyr'
pyr_comps = nest.GetDefaults(pyr_model)["receptor_types"]

intn_model = 'pp_cond_exp_mc_pyr'
intn_comps = nest.GetDefaults(intn_model)["receptor_types"]

syn_ff_pyr_pyr = deepcopy(static_syn_params)
syn_ff_pyr_pyr['receptor_type'] = pyr_comps['basal']

syn_fb_pyr_pyr = deepcopy(static_syn_params)
syn_fb_pyr_pyr['receptor_type'] = pyr_comps['apical_lat']

syn_laminar_pyr_intn = deepcopy(syn_params)
syn_laminar_pyr_intn['receptor_type'] = intn_comps['basal']
# syn_laminar_pyr_intn['eta'] = 0.
syn_laminar_pyr_intn['eta'] = 0.006

syn_laminar_intn_pyr = deepcopy(syn_params)
syn_laminar_intn_pyr['receptor_type'] = pyr_comps['apical_lat']
# syn_laminar_intn_pyr['eta'] = 0.
syn_laminar_intn_pyr['eta'] = 0.0004

# set weights after the fact because deepcopy does not enjoy copying functions.
all_syns = [syn_ff_pyr_pyr, syn_fb_pyr_pyr, syn_laminar_intn_pyr, syn_laminar_pyr_intn]
for s in all_syns:
    s['weight'] = nest.random.uniform(-1, 1)
