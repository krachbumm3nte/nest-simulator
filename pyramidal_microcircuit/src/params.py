import json
from copy import deepcopy

import numpy as np

import nest


class Params:
    def __init__(self, config_file=None):

        # parameters regarding the general simulation environment
        self.delta_t = 0.1  # Euler integration step in ms
        self.threads = 8  # number of threads for parallel processing
        self.record_interval = 1  # interval for storing membrane potentials in ms
        self.t_pres = 50  # stimulus presentation time during training in ms
        self.n_epochs = 1000  # number of training iterations
        self.out_lag = 35  # lag in ms before recording output neuron voltage during testing
        self.target_delay = 0  # lag in ms befor target is presented during training
        self.test_interval = 10  # test the network every N epochs
        self.latent_equilibrium = True  # flag for whether to use latent equilibrium
        self.dims = [9, 30, 3]  # network dimensions, i.e. neurons per layer
        self.init_self_pred = True  # flag to initialize weights to self-predicting state
        self.noise = False  # flag to apply noise to membrane potentials
        self.sigma = 0.3  # standard deviation for membrane potential noise
        self.mode = "bars"  # Which dataset to train on. Default: Bars dataset from Haider (2021)
        self.store_errors = False  # compute and store apical and interneuron errors during traininng
        self.network_type = "snest"
        self.reset = 2  # how to reset the network between simualtions
        self.add_inhibitory_stims = False # add secondary input population to transmit negative inputs in SNN
        # (0: not at all, 1: simulate a relaxation period, 2: hard reset all neuron states)

        # parameters regarding neurons
        self.g_l = 0.03  # somatic leakage conductance
        self.g_a = 0.06  # apical compartment coupling conductance
        self.g_d = 0.1  # basal compartment coupling conductance
        self.g_som = 0.06  # output neuron nudging conductance
        self.g_l_eff = self.g_l + self.g_d + self.g_a  # effective leakage conductance
        self.tau_x = 0.1  # input filtering time constant
        self.g_lk_dnd = self.delta_t  # dendritic leakage
        self.C_m_som = 1  # membrane capacitance of somatic compartment in pF
        self.C_m_bas = 1  # membrane capacitance of basal compartment in pF
        self.C_m_api = 1  # membrane capacitance of apical compartment in pF
        self.psi = 100  # scaling factor for spiking neuron rates
        self.spiking = True  # flag to enable simulation with spiking neurons
        self.t_ref = 0  # refractory period in ms
        # parameters for activation function phi()
        self.gamma = 1
        self.beta = 1
        self.theta = 0

        # parameters for synaptic connections
        self.wmin_init = -1
        self.wmax_init = 1
        self.Wmin = -4
        self.Wmax = 4
        self.tau_delta = 1.  # weight change filter time constant
        self.syn_model = None,  # Synapse model (for NEST simulations only)
        self.p_conn = 1.
        # learning rates for all populations per layer
        self.eta = {
            "ip": [0.004, 0.0],
            "pi": [0.01, 0.0],
            "up": [0.01, 0.003],
            "down": [0.0, 0.0]
        }

        # parameters that regard only simulations in NEST
        self.record_weights = False  # flag to record weights in NEST using a 'weight_recorder'

        # if a config file is provided, read the file and change all specified values
        if config_file:
            self.config_file = config_file
            self.from_json(config_file)

    def setup_nest_configs(self):
        self.neuron_model = 'pp_cond_exp_mc_pyr' if self.spiking else 'rate_neuron_pyr'
        self.syn_model = 'pyr_synapse' if self.spiking else 'pyr_synapse_rate'
        self.static_syn_model = 'static_synapse' if self.spiking else 'rate_connection_delayed'
        self.compartments = nest.GetDefaults(self.neuron_model)["receptor_types"]

        self.pyr_params = {
            'soma': {
                'g_L': self.g_l,
                # misappropriation of somatic conductance. this is the effective somatic leakage conductance now!
                # TODO: create a separate parameter in the neuron model for this
                'g': self.g_l_eff,
                'C_m': self.C_m_som
            },
            'basal': {
                'g_L': self.g_lk_dnd if self.spiking else 1,
                'g': self.g_d,
                'C_m': self.C_m_bas
            },
            'apical_lat': {
                'g_L': self.g_lk_dnd if self.spiking else 1,
                'g': self.g_a,
                'C_m': self.C_m_api
            },
            # 'apical_td': {
            #     'g_L': self.g_lk_dnd if self.spiking else 1,
            #     'g': self.g_a,
            #     'C_m': self.C_m_api
            # },
            'lambda': self.g_som,  # Interneuron nudging conductance
            'gamma': self.gamma,
            'beta': self.beta,
            'theta': self.theta,
            'use_phi': True,  # If False, membrane potentials are transmitted without use of the activation function
            't_ref': self.t_ref,  # refractory period in ms
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
        self.input_params["latent_equilibrium"] = False

        if self.spiking:
            self.input_params["gamma"] = self.psi
            self.pyr_params["gamma"] = self.psi * self.gamma
            self.intn_params["gamma"] = self.psi * self.gamma

            for syn_name in ["ip", "up", "down", "pi"]:
                lr = self.eta[syn_name]
                if syn_name == "pi":
                    self.eta[syn_name] = [eta / (self.psi ** 2 * self.tau_delta) for eta in lr]
                elif syn_name == "down":
                    # TODO: this particular scaling term contains a magic number and is so far unconfirmed
                    self.eta[syn_name] = [2.5 * eta / (self.psi ** 3 * self.tau_delta) for eta in lr]
                else:
                    self.eta[syn_name] = [eta / (self.psi ** 3 * self.tau_delta) for eta in lr]

        self.syn_static = {
            "synapse_model": self.static_syn_model,
            "delay": self.delta_t
        }

        self.syn_plastic = {
            "synapse_model": self.syn_model,
            'tau_Delta': self.tau_delta,
            'Wmin': self.Wmin / (self.psi if self.spiking else 1),
            'Wmax': self.Wmax / (self.psi if self.spiking else 1),
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
        d = dict(sorted(self.to_dict().items()))
        with open(filename, "w") as f:
            json.dump(d, f, indent=4)
