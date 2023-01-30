import nest
from copy import deepcopy
import numpy as np

# Simulation parameters
delta_t = 0.1  # euler integration step in ms
sigma = 0.3  # standard deviation for membrane potential noise

sim_params = {
    "delta_t": delta_t,
    "threads": 10,
    "record_interval": 75,  # interval for storing membrane potentials
    "self_predicting_ff": False,  # initialize feedforward weights to self-predicting state
    "self_predicting_fb": False,  # initialize feedback weights to self-predicting state
    "plasticity": True,  # enable synaptic plasticity
    "SIM_TIME": 100,  # simulation time per input pattern in ms
    "n_runs": 100000,  # number of training iterations
    "noise": True,  # apply noise to membrane potentials
    "sigma": sigma,
    "noise_factor": np.sqrt(delta_t) * sigma,  # constant noise factor for numpy simulations
    "dims": [3, 2, 2],  # network dimensions, i.e. neurons per layer
    "teacher": True,  # If True, teaching current is injected into output layer
    "dims_teacher": [3, 2, 2], # teacher network dimensions.
    "k_yh": 10, # hidden to output teacher weight scaling factor
    "k_hx": 2, # input to hidden teacher weight scaling factor
    "recording_backend": "ascii",  # Backend for NEST multimeter recordings
}


# Neuron parameters
g_l = 0.1  # somatic leakage conductance
g_a = 0.8  # apical compartment coupling conductance
g_d = 1  # basal compartment coupling conductance
lambda_ah = g_a / (g_d + g_a + g_l)  # Useful constant for scaling learning rates
lambda_out = g_d / (g_d + g_l)
g_l_eff = g_l + g_d + g_a


# parameters of the activation function phi()
# gamma = 1
# beta = 1
# theta = 0
gamma = 0.1
beta = 1
theta = 3

neuron_params = {
    "tau_x": 3,  # input filtering time constant
    "g_l": g_l,
    "g_lk_dnd": delta_t,  # dendritic leakage conductance
    "g_a": g_a,
    "g_d": g_d,
    "g_som": 0.8,  # somatic conductance TODO:
    "g_si": 0.8,  # interneuron nudging conductance
    "g_s": 0.8,  # output neuron nudging conductance
    "lambda_out": lambda_out,
    "lambda_ah": lambda_ah,
    'gamma': gamma,
    'beta': beta,
    'theta': theta,
    "g_l_eff": g_l_eff,
    "weight_scale": 150
}


comp_defaults = {
    'V_m': 0.0,  # Membrane potential
    'g_L': neuron_params["g_lk_dnd"],
    'g': neuron_params["g_som"]
}


pyr_params = {
    'soma': deepcopy(comp_defaults),
    'basal': deepcopy(comp_defaults),
    'apical_lat': deepcopy(comp_defaults),
    'tau_m': 1,  # Membrane time constant
    'C_m': 1.0,  # Membrane capacitance
    'lambda': neuron_params["g_si"],  # Interneuron nudging conductance
    'gamma': gamma,
    'beta': beta,
    'theta': theta,
    'use_phi': True,  # If False, rate neuron membrane potentials are transmitted without use of the activation function
    't_ref': 0.
}
# change compartment specific paramters
pyr_params['basal']['g'] = g_d
pyr_params['apical_lat']['g'] = g_a
pyr_params['soma']['g_L'] = g_l
# misappropriation of somatic conductance. this is the effective somatic leakage conductance now!
# TODO: create a separate parameter in the neuron model for this
pyr_params['soma']['g'] = g_l_eff


# Interneurons are effectively pyramidal neurons with a silenced apical compartment.
intn_params = deepcopy(pyr_params)
intn_params['apical_lat']['g'] = 0


# to replace the low pass filtering of the input, input neurons have both
# injected current and leakage conductance attenuated.
# Additionally, the dendritic compartments are silenced, and membrane voltage is
# transmitted to the hidden layer without the nonlinearity phi.
input_params = deepcopy(pyr_params)
input_params["soma"]["g"] = 1/neuron_params["tau_x"]
input_params["basal"]["g"] = 0
input_params["apical_lat"]["g"] = 0
input_params["use_phi"] = False
input_params['tau_m'] = 1/neuron_params["tau_x"]


neuron_params["pyr"] = pyr_params
neuron_params["input"] = input_params
neuron_params["intn"] = intn_params










# Dicts derived from this can be passed directly to nest.Connect() as synapse parameters
tau_delta = 30
syn_params = {
    'synapse_model': None,  # Synapse model (for NEST simulations only)
    'tau_Delta': tau_delta,  # Synaptic time constant
    'Wmin': -10,  # minimum weight
    'Wmax': 10,  # maximum weight
    'delay': sim_params['delta_t'],  # synaptic delay
    'wmin_init': -0.1,  # synaptic weight initialization min
    'wmax_init': 0.1,  # synaptic weight initialization max
    'tau_Delta': tau_delta
}


# connection specific learning rates
if sim_params["plasticity"]:
    # from the mathematica script
    eta_yh = 0.01
    eta_hx = eta_yh / lambda_ah
    eta_ih = 0.01 / lambda_ah
    eta_hi = 5 * eta_ih
    
    # from Sacramento 2018, Fig S1
    # eta_yh = 0
    # eta_ih = 0.0002375 
    # eta_hi = 0.0005
    # eta_hx = 0

    # from Sacramento 2018, Fig 2
    # eta_yh = 0.0005
    # eta_ih = 0.0011875  
    # eta_hi = 0.0005
    # eta_hx = 0.0011875
else:
    eta_yh = 0
    eta_hx = 0
    eta_hi = 0
    eta_ih = 0
eta_hy = 0

for syn_name, eta in zip(["hx", "yh", "hy", "ih", "hi"], [eta_hx, eta_yh, eta_hy, eta_ih, eta_hi]):
    syn_params[syn_name] = {
        'tau_Delta': syn_params["tau_Delta"],
        'Wmin': -10,  # minimum weight
        'Wmax': 10,  # maximum weight
        'eta': eta,
        'delay': 0.1
    }


