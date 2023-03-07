import nest
from copy import deepcopy
import numpy as np

# Simulation parameters
delta_t = 0.1  # euler integration step in ms
sigma = 0.3  # standard deviation for membrane potential noise

sim_params = {
    "delta_t": delta_t,
    "threads": 9,
    "record_interval": 0.2,  # interval for storing membrane potentials in ms
    "init_self_pred": True,  # initialize feedback weights to self-predicting state
    "plasticity": True,  # enable synaptic plasticity
    "SIM_TIME": 10,  # simulation time per input pattern in ms
    "n_epochs": 1000,  # number of training iterations
    "noise": False,  # apply noise to membrane potentials
    "sigma": sigma,
    "noise_factor": np.sqrt(delta_t) * sigma,  # constant noise factor for numpy simulations
    "dims": [9, 30, 3],  # network dimensions, i.e. neurons per layer
    "teacher": True,  # If True, teaching current is injected into output layer
    "dims_teacher": [9, 10, 3],  # teacher network dimensions.
    "k_yh": 10,  # hidden to output teacher weight scaling factor
    "k_hx": 1,  # input to hidden teacher weight scaling factor
    "use_mm": True,  # If true, record activity of nest neurons using multimeters
    "recording_backend": "memory",  # Backend for NEST multimeter recordings
    "out_lag": 4,  # lag in ms before recording output neuron voltage during testing
    "test_interval": 10, # test the network every N epochs 
}


# Neuron parameters
g_l = 0.03  # somatic leakage conductance
g_a = 0.06  # apical compartment coupling conductance
g_d = 0.1  # basal compartment coupling conductance
g_som = 0.06
lambda_ah = g_a / (g_d + g_a + g_l)  # Useful constant for scaling learning rates
lambda_bh = g_d / (g_d + g_a + g_l)  # Useful constant for scaling learning rates

lambda_out = g_d / (g_d + g_l)
g_l_eff = g_l + g_d + g_a


# parameters of the activation function phi()
gamma = 1
beta = 1
theta = 0

# gamma = 0.1
# beta = 1
# theta = 3

neuron_params = {
    "tau_x": 0.1,  # input filtering time constant
    "g_l": g_l,
    "g_lk_dnd": delta_t,  # dendritic leakage conductance
    "g_a": g_a,
    "g_d": g_d,
    "g_som": g_som,  # output neuron nudging conductance
    "lambda_out": lambda_out,
    "lambda_ah": lambda_ah,
    "lambda_bh": lambda_bh,
    'gamma': gamma,
    'beta': beta,
    'theta': theta,
    "g_l_eff": g_l_eff,
    "weight_scale": 250,
    "latent_equilibrium": True
}


comp_defaults = {
    'V_m': 0.0,  # Membrane potential
    'g_L': neuron_params["g_lk_dnd"],
    'g': neuron_params["g_som"],
}


pyr_params = {
    'soma': deepcopy(comp_defaults),
    'basal': deepcopy(comp_defaults),
    'apical_lat': deepcopy(comp_defaults),
    'tau_m': 1,  # Membrane time constant
    'C_m': 1.0,  # Membrane capacitance
    'lambda': neuron_params["g_som"],  # Interneuron nudging conductance
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


neuron_params["pyr"] = pyr_params
neuron_params["input"] = input_params
neuron_params["intn"] = intn_params


Wmin, Wmax = -4, 4
# Dicts derived from this can be passed directly to nest.Connect() as synapse parameters
tau_delta = 1
syn_params = {
    'synapse_model': None,  # Synapse model (for NEST simulations only)
    'tau_Delta': tau_delta,  # Synaptic time constant
    'Wmin': Wmin,  # minimum weight
    'Wmax': Wmax,  # maximum weight
    'delay': sim_params['delta_t'],  # synaptic delay
    'tau_Delta': tau_delta,
    'eta': {
        'ip': [0.02, 0],
        'pi': [0, 0],
        'up': [0.05, 0.01],
    }
}
