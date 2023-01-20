import nest
from copy import deepcopy
import numpy as np

# Simulation parameters
delta_t = 0.1  # euler integration step in ms
sigma = 0.3  # standard deviation for membrane potential noise

sim_params = {
    "delta_t": delta_t,
    "threads": 8,
    "record_interval": 75,  # interval for storing membrane potentials
    "self_predicting_ff": False,  # initialize feedforward weights to self-predicting state
    "self_predicting_fb": False,  # initialize feedback weights to self-predicting state
    "plasticity": True,  # enable synaptic plasticity
    "SIM_TIME": 100,  # simulation time per input pattern in ms
    "n_runs": 10000,  # number of training iterations
    "noise": True,  # apply noise to membrane potentials
    "sigma": sigma,
    "noise_factor": np.sqrt(delta_t) * sigma,  # constant noise factor for numpy simulations
    "dims": [6, 4, 3],  # network dimensions, i.e. neurons per layer
    "recording_backend": "ascii", # Backend for NEST multimeter recordings
    "teacher": True, # If True, teaching current is injected into output layer 
}


# Neuron parameters
# TODO: add units maybe?
g_l = 0.1  # somatic leakage conductance
g_a = 0.8  # apical compartment coupling conductance
g_d = 1  # basal compartment coupling conductance
lambda_ah = g_a / (g_d + g_a + g_l)  # Useful constant for scaling learning rates

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
    "lambda_ah": lambda_ah,
    'gamma': gamma,
    'beta': beta,
    'theta': theta,
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
pyr_params['soma']['g'] = g_l + g_d + neuron_params["g_si"]


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

# connection specific learning rates
# TODO: clean this up!
if sim_params["plasticity"]:
    eta_yh = 0.01
    eta_hx = eta_yh / lambda_ah
    eta_ih = 0.01 / lambda_ah
    eta_hi = 5 * eta_ih    
    # eta_yh = 0.01 * 0.5
    # eta_hx = eta_yh / lambda_ah * 0.5
    # eta_ih = 0.01 / lambda_ah * 0.5
    # eta_hi = 5 * eta_ih * 0.5
    # eta_yh = 0
    # eta_ih = 0.0002375  # from Sacramento, Fig S1
    # eta_hi = 0.0005
    # eta_hx = 0
else:
    eta_yh = 0
    eta_hx = 0
    eta_hi = 0
    eta_ih = 0
eta_hy = 0


# Dicts derived from this can be passed directly to nest.Connect() as synapse parameters
syn_defaults = {
    'synapse_model': None,  # Synapse model (for NEST simulations only)
    'tau_Delta': 30,  # Synaptic time constant
    'Wmin': -10,  # minimum weight
    'Wmax': 10,  # maximum weight
    'eta': 0.0,  # learning rate
    'delay': sim_params['delta_t'],  # synaptic delay
}

syn_params = deepcopy(syn_defaults)
syn_params.update({
    'wmin_init': -1,  # synaptic weight initialization min
    'wmax_init': 1,  # synaptic weight initialization max
})


def setup_models(spiking, record_weights=False):

    wr = None

    neuron_model = 'pp_cond_exp_mc_pyr' if spiking else 'rate_neuron_pyr'
    pyr_comps = nest.GetDefaults(neuron_model)["receptor_types"]
    neuron_params["model"] = neuron_model

    basal_dendrite = pyr_comps['basal']
    apical_dendrite = pyr_comps['apical_lat']

    syn_model = 'pyr_synapse' if spiking else 'pyr_synapse_rate'

    if not spiking:
        neuron_params["pyr"]["basal"]["g_L"] = 1
        neuron_params["pyr"]["apical_lat"]["g_L"] = 1
        neuron_params["intn"]["basal"]["g_L"] = 1
        neuron_params["intn"]["apical_lat"]["g_L"] = 1
        neuron_params["input"]["basal"]["g_L"] = 1
        neuron_params["input"]["apical_lat"]["g_L"] = 1
        

    if record_weights:
        wr = nest.Create("weight_recorder")
        nest.CopyModel(syn_model, 'record_syn', {"weight_recorder": wr})
        syn_model = 'record_syn'

    syn_defaults["synapse_model"] = syn_model

    syn_params["hx"] = deepcopy(syn_defaults)
    syn_params["hx"]['receptor_type'] = basal_dendrite
    syn_params["hx"]['eta'] = eta_hx

    syn_params["yh"] = deepcopy(syn_defaults)
    syn_params["yh"]['receptor_type'] = basal_dendrite
    syn_params["yh"]['eta'] = eta_yh

    syn_params["hy"] = deepcopy(syn_defaults)
    syn_params["hy"]['receptor_type'] = apical_dendrite
    syn_params["hy"]['eta'] = eta_hy

    syn_params["ih"] = deepcopy(syn_defaults)
    syn_params["ih"]['receptor_type'] = basal_dendrite
    syn_params["ih"]['eta'] = eta_ih

    syn_params["hi"] = deepcopy(syn_defaults)
    syn_params["hi"]['receptor_type'] = apical_dendrite
    syn_params["hi"]['eta'] = eta_hi

    return wr


def phi(x):
    return gamma * np.log(1 + np.exp(beta * (x - theta)))


def phi_constant(x):
    return np.log(1.0 + np.exp(x))


def phi_inverse(x):
    return (1 / beta) * (beta * theta + np.log(np.exp(x/gamma) - 1))


neuron_params["phi"] = phi
neuron_params["phi_inverse"] = phi_inverse
