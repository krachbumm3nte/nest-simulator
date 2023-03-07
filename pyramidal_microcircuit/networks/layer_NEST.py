
import numpy as np
from abc import ABC, abstractmethod
from .layer import AbstractLayer
import nest
import time
dtype = np.float32


class AbstractLayer():
    def __init__(self, nrn, sim, syn, eta) -> None:
        self.ga = nrn["g_a"]
        self.gb = nrn["g_d"]  # TODO: separate these?
        self.gd = nrn["g_d"]
        self.gl = nrn["g_l"]
        self.gsom = nrn["g_som"]
        self.tau_x = nrn["tau_x"]
        self.le = nrn["latent_equilibrium"]

        self.noise_factor = sim["noise_factor"] if sim["noise"] else 0
        self.tau_delta = syn["tau_Delta"]
        self.dt = sim["delta_t"]

        self.leakage = self.gl + self.ga + self.gb
        self.lambda_out = nrn["lambda_out"]
        self.lambda_ah = nrn["lambda_ah"]
        self.lambda_bh = nrn["lambda_bh"]
        self.eta = eta.copy()

        self.Wmin = -4
        self.Wmax = 4  # TODO: read from config
        self.gamma = nrn["gamma"]
        self.beta = nrn["beta"]
        self.theta = nrn["theta"]
        self.weight_scale = nrn["weight_scale"] if sim["spiking"] else 1

    @abstractmethod
    def update(self, r_in, u_next, plasticity, noise_on=False):
        pass

    @abstractmethod
    def apply(self, plasticity):
        pass

    @abstractmethod
    def reset(self, reset_weights=False):
        pass

    def gen_weights(self, n_in, n_out, wmin=None, wmax=None):
        if not wmin:
            wmin = -1/self.weight_scale
        if not wmax:
            wmax = 1/self.weight_scale
        return np.random.uniform(wmin, wmax, (n_out, n_in))

    def phi(self, x, thresh=15):

        res = x.copy()
        ind = np.abs(x) < thresh
        res[x < -thresh] = 0
        res[ind] = np.log(1 + np.exp(x[ind]))
        return res
        return self.gamma * np.log(1 + np.exp(self.beta * (x - self.theta)))


class NestLayer(AbstractLayer):

    def __init__(self, nrn, sim, syn, n, eta, pyr_prev, intn_prev=None) -> None:
        super().__init__(nrn, sim, syn, eta)

        self.N_in = sim["dims"][n]
        self.N_pyr = sim["dims"][n+1]
        self.N_next = sim["dims"][n+2]

        self.syn_up = syn["conns"][n]["up"]
        self.syn_pi = syn["conns"][n]["pi"]
        self.syn_ip = syn["conns"][n]["ip"]
        self.syn_down = syn["conns"][n]["down"]

        self.syn_up["weight"] = self.gen_weights(self.N_in, self.N_pyr)
        self.syn_pi["weight"] = self.gen_weights(self.N_next, self.N_pyr)
        self.syn_ip["weight"] = self.gen_weights(self.N_pyr, self.N_next)
        self.syn_down["weight"] = self.gen_weights(self.N_next, self.N_pyr)

        self.pyr = nest.Create(nrn["model"], self.N_pyr, nrn["pyr"])
        self.intn = nest.Create(nrn["model"], self.N_next, nrn["intn"])

        self.up = nest.Connect(pyr_prev, self.pyr, syn_spec=self.syn_up, return_synapsecollection=True)
        self.pi = nest.Connect(self.intn, self.pyr, syn_spec=self.syn_pi, return_synapsecollection=True)
        self.ip = nest.Connect(self.pyr, self.intn, syn_spec=self.syn_ip, return_synapsecollection=True)

        if intn_prev is not None:
            for i in range(len(self.pyr)):
                self.pyr[i].target = intn_prev[i].get("global_id")

    def set_feedback_conns(self, pyr_next):
        self.down = nest.Connect(pyr_next, self.pyr, syn_spec=self.syn_down, return_synapsecollection=True)

    def reset(self, reset_weights=False):
        '''
        Reset all potentials and Deltas (weight update matrices) to zero.
        Parameters
        ----------
        reset_weights:  Also draw weights again from random distribution.

        '''
        self.pyr.set({"soma": {"V_m": 0, "I_e": 0}, "basal": {"V_m": 0, "I_e": 0}, "apical_lat": {"V_m": 0, "I_e": 0}})
        self.intn.set({"soma": {"V_m": 0, "I_e": 0}, "basal": {"V_m": 0, "I_e": 0}, "apical_lat": {"V_m": 0, "I_e": 0}})

        # TODO: reset urbanczik History?

    def redefine_connections(self, pyr_prev, pyr_next):
        self.up = nest.GetConnections(pyr_prev, self.pyr)
        self.pi = nest.GetConnections(self.intn, self.pyr)
        self.ip = nest.GetConnections(self.pyr, self.intn)
        self.down = nest.GetConnections(pyr_next, self.pyr)

    def enable_learning(self):
        for conn_type in ["up", "pi", "ip"]:
            if self.eta[conn_type] > 0:
                eval(f"self.{conn_type}").set({"eta": self.eta[conn_type]})


class NestOutputLayer(AbstractLayer):

    def __init__(self, nrn, sim, syn, eta, pyr_prev, intn_prev) -> None:
        super().__init__(nrn, sim, syn, eta)

        self.ga = 0
        self.N_in = sim["dims"][-2]
        self.N_out = sim["dims"][-1]

        syn_up = syn["conns"][-1]["up"]
        syn_up["weight"] = self.gen_weights(self.N_in, self.N_out)

        self.pyr = nest.Create(nrn["model"], self.N_out, nrn["pyr"])
        self.pyr.set({"apical_lat": {"g": 0}})
        self.up = nest.Connect(pyr_prev, self.pyr, syn_spec=syn_up, return_synapsecollection=True)
        for i in range(len(self.pyr)):
            self.pyr[i].target = intn_prev[i].get("global_id")

    def reset(self, reset_weights=False):
        '''
        Reset all potentials and Deltas (weight update matrices) to zero.
        Parameters
        ----------
        reset_weights:  Also draw weights again from random distribution.

        '''
        self.pyr.set({"soma": {"V_m": 0, "I_e": 0}, "basal": {"V_m": 0, "I_e": 0}, "apical_lat": {"V_m": 0, "I_e": 0}})

        # TODO: reset urbanczik History?

    def redefine_connections(self, pyr_prev):
        self.up = nest.GetConnections(pyr_prev, self.pyr)

    def enable_learning(self):
        if self.eta["up"] > 0:
            self.up.set({"eta": self.eta["up"]})
