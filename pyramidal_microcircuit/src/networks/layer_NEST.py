
from copy import deepcopy

from src.networks.layer import AbstractLayer
import numpy as np
import nest


class NestLayer(AbstractLayer):

    def __init__(self, net, p, layer, init_weights=None) -> None:
        super().__init__(p, net, layer)
        self.synapses = {}
        for type in ["up", "pi", "ip", "down"]:
            eta = self.eta[type]

            if eta != 0:
                self.synapses[type] = deepcopy(p.syn_plastic)
                self.synapses[type]["eta"] = eta
            else:
                self.synapses[type] = deepcopy(p.syn_static)

        basal_dendrite = p.compartments['basal']
        # apical_distal = p.compartments['apical_td']
        apical_proximal = p.compartments['apical_lat']
        self.synapses["up"]['receptor_type'] = basal_dendrite
        self.synapses["ip"]['receptor_type'] = basal_dendrite
        self.synapses["pi"]['receptor_type'] = apical_proximal
        self.synapses["down"]['receptor_type'] = apical_proximal  # apical_distal

        self.N_prev = net.dims[layer]
        self.N_pyr = net.dims[layer+1]
        self.N_next = net.dims[layer+2]

        if init_weights:
            for type in ["up", "pi", "ip", "down"]:
                # Init weights are assumed to be normalized and thus need to be scaled down.
                self.synapses[type]["weight"] = np.array(init_weights[type]) / self.weight_scale
        else:
            self.synapses["up"]["weight"] = self.gen_weights(self.N_prev, self.N_pyr)
            self.synapses["pi"]["weight"] = self.gen_weights(self.N_next, self.N_pyr)
            self.synapses["ip"]["weight"] = self.gen_weights(self.N_pyr, self.N_next)
            self.synapses["down"]["weight"] = self.gen_weights(self.N_next, self.N_pyr)

        self.pyr = nest.Create(p.neuron_model, self.N_pyr, p.pyr_params)
        self.intn = nest.Create(p.neuron_model, self.N_next, p.intn_params)

    def connect(self, pyr_prev, pyr_next, intn_prev=None):
        self.up = nest.Connect(pyr_prev, self.pyr, syn_spec=self.synapses["up"], return_synapsecollection=True)
        self.pi = nest.Connect(self.intn, self.pyr, syn_spec=self.synapses["pi"], return_synapsecollection=True)
        self.ip = nest.Connect(self.pyr, self.intn, syn_spec=self.synapses["ip"], return_synapsecollection=True)
        self.down = nest.Connect(pyr_next, self.pyr, syn_spec=self.synapses["down"], return_synapsecollection=True)

        if intn_prev is not None:
            for i in range(len(self.pyr)):
                self.pyr[i].target = intn_prev[i].get("global_id")

    def reset(self):
        '''
        Reset all potentials and Deltas (weight update matrices) to zero.
        '''
        self.pyr.set({"soma": {"V_m": 0, "I_e": 0},
                      "basal": {"V_m": 0, "I_e": 0},
                      "apical_lat": {"V_m": 0, "I_e": 0}})
        # "apical_td": {"V_m": 0, "I_e": 0}})
        self.intn.set({"soma": {"V_m": 0, "I_e": 0},
                       "basal": {"V_m": 0, "I_e": 0},
                       "apical_lat": {"V_m": 0, "I_e": 0}})
        # "apical_td": {"V_m": 0, "I_e": 0}})

        # TODO: reset urbanczik History?

    def redefine_connections(self, pyr_prev, pyr_next):
        self.up = nest.GetConnections(pyr_prev, self.pyr)
        self.pi = nest.GetConnections(self.intn, self.pyr)
        self.ip = nest.GetConnections(self.pyr, self.intn)
        self.down = nest.GetConnections(pyr_next, self.pyr)

    def enable_learning(self):
        for conn_type in ["up", "pi", "ip", "down"]:
            if self.eta[conn_type] > 0:
                eval(f"self.{conn_type}").set({"eta": self.eta[conn_type]})


class NestOutputLayer(AbstractLayer):

    def __init__(self, net, p, init_weights=None) -> None:
        super().__init__(p, net, len(net.dims)-2)

        self.synapses = {}
        self.ga = 0
        self.N_prev = net.dims[-2]
        self.N_out = net.dims[-1]


        if self.eta["up"] > 0:
            self.synapses["up"] = deepcopy(p.syn_plastic)
            self.synapses["up"]["eta"] = self.eta["up"]
        else:
            self.synapses["up"] = deepcopy(p.syn_static)

        self.synapses["up"]['receptor_type'] = p.compartments['basal']
        if init_weights:
            self.synapses["up"]["weight"] = np.array(init_weights["up"]) / self.weight_scale
        else:
            self.synapses["up"]["weight"] = self.gen_weights(self.N_prev, self.N_out)

        self.pyr = nest.Create(p.neuron_model, self.N_out, p.pyr_params)
        self.pyr.set({"apical_lat": {"g": 0}})

    def connect(self, pyr_prev, intn_prev):
        self.up = nest.Connect(pyr_prev, self.pyr, syn_spec=self.synapses["up"], return_synapsecollection=True)

        for i in range(len(self.pyr)):
            self.pyr[i].target = intn_prev[i].get("global_id")

    def reset(self):
        '''
        Reset all potentials and to zero.
        '''
        self.pyr.set({"soma": {"V_m": 0, "I_e": 0},
                      "basal": {"V_m": 0, "I_e": 0},
                      "apical_lat": {"V_m": 0, "I_e": 0}})
        # "apical_td": {"V_m": 0, "I_e": 0}})

        # TODO: reset urbanczik History?

    def redefine_connections(self, pyr_prev):
        self.up = nest.GetConnections(pyr_prev, self.pyr)

    def enable_learning(self):
        if self.eta["up"] > 0:
            self.up.set({"eta": self.eta["up"]})
