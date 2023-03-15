import nest
import numpy as np
import json
from copy import deepcopy


class Params:
    def __init__(self, config_file=None):

        # parameters regarding the general simulation environment
        self.delta_t = 0.1         # Euler integration step in ms
        self.threads = 10         # number of threads for parallel processing
        self.record_interval = 0.5         # interval for storing membrane potentials in ms
        # simulation time per input pattern in ms #TODO: un-capitalize
        self.SIM_TIME = 100
        self.n_epochs = 1000         # number of training iterations
        self.out_lag = 75         # lag in ms before recording output neuron voltage during testing
        self.test_interval = 25         # test the network every N epochs
        # flag for whether to use latent equilibrium during training
        self.latent_equilibrium = False
        # network dimensions, i.e. neurons per layer
        self.dims = [9, 30, 3]
        # flag to initialize feedback weights to self-predicting state
        self.init_self_pred = True
        self.noise = False         # flag to apply noise to membrane potentials
        self.sigma = 0.3         # standard deviation for membrane potential noise
        # constant noise factor for numpy simulations
        self.noise_factor = np.sqrt(self.delta_t) * self.sigma
        # Which dataset to train on. Default: Bars dataset from Haider (2021)
        self.mode = "bars"

        # parameters regarding neurons
        self.g_l = 0.03         # somatic leakage conductance
        self.g_a = 0.06         # apical compartment coupling conductance
        self.g_d = 0.1         # basal compartment coupling conductance
        self.g_som = 0.06         # output neuron nudging conductance
        # effective leakage conductance
        self.g_l_eff = self.g_l + self.g_d + self.g_a
        self.tau_x = 0.1        # input filtering time constant
        self.tau_m = 2  # membrane time constant for pyramidal and interneurons
        self.g_lk_dnd = self.delta_t        # dendritic leakage conductance
        # Useful constants for scaling learning rates
        self.lambda_ah = self.g_a / (self.g_d + self.g_a + self.g_l)
        self.lambda_bh = self.g_d / (self.g_d + self.g_a + self.g_l)
        self.lambda_out = self.g_d / (self.g_d + self.g_l)

        # parameters for activation function phi()
        self.gamma = 1
        self.beta = 1
        self.theta = 0

        # parameters for synaptic connections
        self.Wmin = -4
        self.Wmax = 4
        self.tau_delta = 2
        self.synapse_model = None,  # Synapse model (for NEST simulations only)
        self.eta = {
            'ip': [0.002, 0],
            'pi': [0, 0],
            'up': [0.005, 0.001],
            # 'ip': [0.0001, 0],
            # 'pi': [0, 0],
            # 'up': [0.00025, 0.00005],
            'down': [0, 0]
        }

        # parameters that regard only simulations in NEST
        # flag to record weights in NEST using a 'weight_recorder'
        self.record_weights = False
        self.weight_scale = 150        # weight scaling factor # TODO: rename this
        self.spiking = True        # flag to enable simulation with spiking neurons

        if config_file:
            self.from_json(config_file)

    def setup_nest_configs(self):
        self.neuron_model = 'pp_cond_exp_mc_pyr' if self.spiking else 'rate_neuron_pyr'
        self.syn_model = 'pyr_synapse' if self.spiking else 'pyr_synapse_rate'
        self.static_syn_model = 'static_synapse' if self.spiking else 'rate_connection_delayed'
        self.compartments = nest.GetDefaults(self.neuron_model)[
            "receptor_types"]

        self.pyr_params = {
            'soma': {
                'g_L': self.g_l,
                # misappropriation of somatic conductance. this is the effective somatic leakage conductance now!
                # TODO: create a separate parameter in the neuron model for this
                'g': self.g_l_eff,
            },
            'basal': {
                'g_L': self.g_lk_dnd if self.spiking else 1,
                'g': self.g_d,
            },
            'apical_lat': {
                'g_L': self.g_lk_dnd if self.spiking else 1,
                'g': self.g_a,
            },
            'tau_m': self.tau_m,  # Membrane time constant
            'C_m': 1.0,  # Membrane capacitance
            'lambda': self.g_som,  # Interneuron nudging conductance
            'gamma': self.gamma,
            'beta': self.beta,
            'theta': self.theta,
            'use_phi': True,  # If False, rate neuron membrane potentials are transmitted without use of the activation function
            't_ref': 0.,
            'latent_equilibrium': self.latent_equilibrium
        }

        # Interneurons are effectively pyramidal neurons with a silenced apical compartment.
        self.intn_params = deepcopy(self.pyr_params)
        self.intn_params['apical_lat']['g'] = 0

        # to replace the low pass filtering of the input, input neurons have both
        # injected current and leakage conductance attenuated.
        # Additionally, the dendritic compartments are silenced, and membrane voltage is
        # transmitted to the hidden layer without the nonlinearity phi.
        self.input_params = deepcopy(self.pyr_params)
        self.input_params["soma"]["g"] = 1/self.tau_x
        self.input_params["basal"]["g"] = 0
        self.input_params["apical_lat"]["g"] = 0
        self.input_params["use_phi"] = False

        if self.spiking:
            self.input_params["gamma"] = self.weight_scale
            self.pyr_params["gamma"] = self.weight_scale * self.gamma
            self.intn_params["gamma"] = self.weight_scale * self.gamma

            for syn_name in ["ip", "up", "down", "pi"]:
                lr = self.eta[syn_name]
                if syn_name == "pi":
                    self.eta[syn_name] = [eta / self.weight_scale **
                                          2 * self.tau_delta for eta in lr]
                else:
                    self.eta[syn_name] = [eta / self.weight_scale **
                                          3 * self.tau_delta for eta in lr]

        self.syn_static = {
            "synapse_model": self.static_syn_model,
            "delay": self.delta_t
        }

        self.syn_plastic = {
            "synapse_model": self.syn_model,
            'tau_Delta': self.tau_delta,
            # minimum weight
            'Wmin': self.Wmin / (self.weight_scale if self.spiking else 1),
            # maximum weight
            'Wmax': self.Wmax / (self.weight_scale if self.spiking else 1),
            'delay': self.delta_t
        }

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}

    def from_dict(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)

    def from_json(self, filename):
        with open(filename, "r") as f:
            self.from_dict(json.load(f))

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
