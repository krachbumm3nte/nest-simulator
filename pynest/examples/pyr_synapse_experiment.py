import nest
import matplotlib.pyplot as plt
import numpy as np

resolution = 0.1
nest.resolution = resolution

# neuron parameters
nrn_model = 'pp_cond_exp_mc_pyr'
nrn_params = {
    't_ref': 3.0,        # refractory period
    'g_sp': 0.0,       # soma-to-dendritic coupling conductance
    'g_ps': 600.,
    'soma': {
        'V_m': -70.0,    # initial value of V_m
        'C_m': 300.0,    # capacitance of membrane
        'E_L': -70.0,    # resting potential
        'g_L': 30.0,     # somatic leak conductance
        'E_ex': 0.0,     # resting potential for exc input
        'E_in': -75.0,   # resting potential for inh input
        'tau_syn_ex': 3.0,  # time constant of exc conductance
        'tau_syn_in': 3.0,  # time constant of inh conductance
    },
    'basal': {
        'V_m': -70.0,    # initial value of V_m
        'C_m': 300.0,    # capacitance of membrane
        'E_L': -70.0,    # resting potential
        'g_L': 30.0,     # dendritic leak conductance
        'tau_syn_ex': 3.0,  # time constant of exc input current
        'tau_syn_in': 3.0,  # time constant of inh input current
    },
    # parameters of rate function
    'phi_max': 0.15,     # max rate
    'rate_slope': 0.5,   # called 'k' in the paper
    'beta': 1.0 / 3.0,
    'theta': -55.0,
}

# synapse params
syns = nest.GetDefaults(nrn_model)['receptor_types']
print(syns)
init_w = 2.8 * nrn_params['basal']['C_m']
syn_params = {
    'synapse_model': 'urbanczik_synapse',
    'receptor_type': syns['basal_exc'],
    'tau_Delta': 100.0,  # time constant of low pass filtering of the weight change
    'eta': 0.17,         # learning rate
    'weight': init_w,
    'Wmax': 4.5 * nrn_params['basal']['C_m'],
    'delay': resolution,
}


"""
neuron and devices
"""

b_stim = nest.Create("parrot_neuron")
a_stim = nest.Create("parrot_neuron")

post = nest.Create(nrn_model, params=nrn_params)

nest.Connect(b_stim, post, syn_spec=syn_params)
print(syn_params["receptor_type"])
syn_params["receptor_type"] = syns['apical_exc']
nest.Connect(a_stim, post, syn_spec=syn_params)
print(syn_params["receptor_type"])



b_p = nest.Create("poisson_generator")
a_p = nest.Create("poisson_generator")
nest.Connect(b_p, b_stim)
nest.Connect(a_p, a_stim)

mm = nest.Create('multimeter', 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a"]})
#nest.Connect(mm, post)

b_p.rate = 1000
nest.Simulate(200)
b_p.rate = 0
a_p.rate = 1000
nest.Simulate(200)

foo = mm.get("events")


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.plot(foo["V_m.s"])
ax2.plot(foo["V_m.b"])
ax3.plot(foo["V_m.a"])
plt.show()