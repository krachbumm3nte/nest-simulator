import nest
import numpy as np
from .network import Network
from sklearn.metrics import mean_squared_error as mse
from time import time
import pandas as pd
from copy import deepcopy
from .layer_NEST import NestLayer, NestOutputLayer
from .params import Params


class NestNetwork(Network):

    def __init__(self, p: Params) -> None:
        super().__init__(p)

        self.noise_generator = None
        self.spiking = p.spiking
        self.weight_scale = self.p.weight_scale if self.spiking else 1
        self.recording_backend = "memory"         # backend for NEST multimeter recordings
        self.use_mm = True         # flag to record activity of nest neurons using multimeters

        self.setup_populations()

        self.iteration = 0

    def setup_populations(self):

        self.wr = None
        if self.p.record_weights:
            self.wr = nest.Create("weight_recorder", params={'record_to': "ascii", "precision": 12})
            # wr = nest.Create("weight_recorder")
            nest.CopyModel(self.p.syn_model, 'record_syn', {"weight_recorder": self.wr})
            self.p.syn_model = 'record_syn'
            nest.CopyModel(self.p.static_syn_model, 'static_record_syn', {"weight_recorder": self.wr})
            self.p.static_syn_model = 'static_record_syn'
        self.p.setup_nest_configs()
        # Create input layer neurons
        self.input_neurons = nest.Create(self.p.neuron_model, self.dims[0], self.p.input_params)

        pyr_prev = self.input_neurons
        intn_prev = None
        for i in range(len(self.dims)-2):

            layer = NestLayer(self, self.p, i)
            self.layers.append(layer)
            pyr_prev = layer.pyr
            intn_prev = layer.intn

        self.layers.append(NestOutputLayer(self, self.p))

        if self.p.noise:
            # Inject Gaussian white noise into neuron somata.
            self.noise_generator = nest.Create("noise_generator", 1, {"mean": 0., "std": self.sigma_noise})
            for l_current in self.layers[:-1]:
                nest.Connect(self.noise_generator, l_current.pyr, syn_spec={"receptor_type": self.p.compartments["soma_curr"]})
                nest.Connect(self.noise_generator, l_current.intn, syn_spec={"receptor_type": self.p.compartments["soma_curr"]})
            nest.Connect(self.noise_generator, self.layers[-1].pyr, syn_spec={"receptor_type": self.p.compartments["soma_curr"]})

        if self.use_mm:
            self.mm = nest.Create('multimeter', 1, {'record_to': self.recording_backend,
                                                    'interval': self.p.record_interval,
                                                    'record_from': ["V_m.a_lat", "V_m.s", "V_m.b"],
                                                    'stop': 0.0  # disables multimeter by default
                                                    })
            nest.Connect(self.mm, self.input_neurons)
            nest.Connect(self.mm, self.layers[0].pyr)
            nest.Connect(self.mm, self.layers[0].intn)
            nest.Connect(self.mm, self.layers[-1].pyr)
        # TODO: keep both ways of storage?
        self.U_y_record = np.zeros((1, self.dims[-1]))
        self.V_ah_record = np.zeros((1, self.dims[1]))
        self.U_h_record = np.zeros((1, self.dims[1]))
        self.U_i_record = np.zeros((1, self.dims[-1]))

        # step generators for enabling batch training
        self.sgx = nest.Create("step_current_generator", self.dims[0])
        nest.Connect(self.sgx, self.input_neurons, conn_spec='one_to_one',
                     syn_spec={"receptor_type": self.p.compartments["soma_curr"]})
        self.sgy = nest.Create("step_current_generator", self.dims[-1])
        nest.Connect(self.sgy, self.layers[-1].pyr, conn_spec='one_to_one',
                     syn_spec={"receptor_type": self.p.compartments["soma_curr"]})

        pyr_prev = self.input_neurons
        intn_prev = None
        for i in range(len(self.dims)-2):
            layer = self.layers[i]
            pyr_next = self.layers[i+1].pyr
            layer.connect(pyr_prev, pyr_next, intn_prev)
            pyr_prev = layer.pyr
            intn_prev = layer.intn
        self.layers[-1].connect(pyr_prev, intn_prev)

        pyr_prev = self.input_neurons
        for i in range(len(self.layers)-1):
            self.layers[i].redefine_connections(pyr_prev, self.layers[i+1].pyr)
            pyr_prev = self.layers[i].pyr
        self.layers[-1].redefine_connections(pyr_prev)

        for i in range(len(self.layers) - 1):
            l_current = self.layers[i]
            l_next = self.layers[i + 1]
            if self.p.init_self_pred:
                w_down = self.get_weight_array_from_syn(l_current.down)
                self.set_weights_from_syn(-w_down, l_current.pi)
                w_up = self.get_weight_array_from_syn(l_next.up)

                self.set_weights_from_syn(w_up * l_next.gb / (l_next.gl + l_next.ga + l_next.gb) *
                                          (l_current.gl + l_current.gd) / l_current.gd, l_current.ip)
        
        
    def simulate(self, T, enable_recording=False):
        if enable_recording:
            # TODO: record with out_lag aswell?
            self.mm.set({"start": 0, 'stop': self.sim_time, 'origin': nest.biological_time})
            if self.recording_backend == "ascii":
                nest.SetKernelStatus({"data_prefix": f"it{str(self.iteration).zfill(8)}_"})

        nest.Simulate(T)
        self.iteration += 1

    def set_input(self, input_currents):
        """Inject a constant current into all neurons in the input layer.

        @note: Before injection, currents are attenuated by the input time constant
        in order to match synaptic filtering.

        Arguments:
            input_currents -- Iterable of length equal to the input dimension.
        """
        self.input_currents = input_currents
        for i in range(self.dims[0]):
            self.input_neurons[i].set({"soma": {"I_e": input_currents[i] / self.p.tau_x}})

    def train_batch(self, x_batch, y_batch):
        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            self.set_input(x)
            self.set_target(y)
            self.simulate(self.sim_time)
            # if i == len(x)-1:
            #     U_y = [nrn.get("soma")["V_m"] for nrn in self.layers[-1].pyr]
            #     if not self.use_mm:
            #         U_h = [nrn.get("soma")["V_m"] for nrn in self.layers[0].pyr]
            #         V_ah = [nrn.get("apical_lat")["V_m"] for nrn in self.layers[0].pyr]
            #         U_i = [nrn.get("soma")["V_m"] for nrn in self.layers[0].intn]

            #         self.V_ah_record = np.concatenate((self.V_ah_record, np.expand_dims(V_ah, 0)), axis=0)
            #         self.U_h_record = np.concatenate((self.U_h_record, np.expand_dims(U_h, 0)), axis=0)
            #         self.U_i_record = np.concatenate((self.U_i_record, np.expand_dims(U_i, 0)), axis=0)
            #         self.U_y_record = np.concatenate((self.U_y_record, np.expand_dims(U_y, 0)), axis=0)

            #     self.train_loss.append((self.epoch, mse(y, U_y)))
            self.reset()

    def test_batch(self, x_batch, y_batch):
        acc = []
        loss_mse = []
        # set all learning rates to zero
        self.disable_learning()

        for x_test, y_actual in zip(x_batch, y_batch):
            self.set_input(x_test)
            self.mm.set({"start": self.p.out_lag, 'stop': self.sim_time, 'origin': nest.biological_time})
            self.simulate(self.sim_time)
            mm_data = pd.DataFrame.from_dict(self.mm.events)
            U_Y = [mm_data[mm_data["senders"] == out_id]["V_m.s"] for out_id in self.layers[-1].pyr.global_id]
            y_pred = np.mean(U_Y, axis=1)
            loss_mse.append(mse(y_actual, y_pred))
            acc.append(np.argmax(y_actual) == np.argmax(y_pred))
            self.reset()

        # set learning rates to their original values
        self.enable_learning()
        return np.mean(acc), np.mean(loss_mse)

    def disable_learning(self):
        nest.GetConnections(synapse_model=self.p.syn_model).set({"eta": 0})

    def enable_learning(self):
        for l in self.layers:
            l.enable_learning()

    def set_target(self, target_currents):
        """Inject a constant current into all neurons in the output layer.

        @note: Before injection, currents are attenuated by the output neuron
        nudging conductance in order to match the simulation exactly.

        Arguments:
            target_currents -- Iterable of length equal to the output dimension.
        """
        self.target_curr = target_currents
        for i in range(self.dims[-1]):
            self.layers[-1].pyr[i].set({"soma": {"I_e": target_currents[i] * self.p.g_som}})

    def get_weight_array(self, source, target, normalized=False):
        weight_df = pd.DataFrame.from_dict(nest.GetConnections(source=source, target=target).get())
        weight_array = weight_df.sort_values(["target", "source"]).weight.values.reshape((len(target), len(source)))
        if normalized:
            weight_array *= self.weight_scale
        return weight_array

    def get_weight_array_from_syn(self, synapse_collection, normalized=False):
        weight_df = pd.DataFrame.from_dict(synapse_collection.get())
        n_out = len(set(synapse_collection.targets()))
        n_in = len(set(synapse_collection.sources()))
        weight_array = weight_df.sort_values(["target", "source"]).weight.values.reshape((n_out, n_in))
        if normalized:
            weight_array *= self.weight_scale
        return weight_array

    def get_weight_dict(self, normalized=True):
        weights = []
        for l in self.layers[:-1]:
            weights.append({"up": self.get_weight_array_from_syn(l.up, normalized),
                            "pi": self.get_weight_array_from_syn(l.pi, normalized),
                            "ip": self.get_weight_array_from_syn(l.ip, normalized),
                            "down": self.get_weight_array_from_syn(l.down, normalized)})
        weights.append({"up": self.get_weight_array_from_syn(self.layers[-1].up, normalized)})
        return weights

    def reset(self):
        self.input_neurons.set({"soma": {"V_m": 0, "I_e": 0}, "basal": {
                               "V_m": 0, "I_e": 0}, "apical_lat": {"V_m": 0, "I_e": 0}})
        for l in self.layers:
            l.reset()
        self.mm.n_events = 0

    def set_weights_from_syn(self, weights, synapse_collection):
        # TODO: match numpy variant
        for i, source_id in enumerate(sorted(set(synapse_collection.sources()))):
            for j, target_id in enumerate(sorted(set(synapse_collection.targets()))):
                source = nest.GetNodes({"global_id": source_id})
                target = nest.GetNodes({"global_id": target_id})
                nest.GetConnections(source, target).set({"weight": weights[j][i]})

    def set_all_weights(self, weight_dict):
        for i, layer in enumerate(self.layers[:-1]):
            self.set_weights_from_syn(weight_dict[i]["up"], layer.up)
            self.set_weights_from_syn(weight_dict[i]["ip"], layer.ip)
            self.set_weights_from_syn(weight_dict[i]["pi"], layer.pi)
            self.set_weights_from_syn(weight_dict[i]["down"], layer.down)
        self.set_weights_from_syn(weight_dict[-1]["up"], self.layers[-1].up)
