import nest
import numpy as np
from .network import Network
from sklearn.metrics import mean_squared_error as mse
from time import time
import pandas as pd
from copy import deepcopy
from .layer_NEST import NestLayer, NestOutputLayer


class NestNetwork(Network):

    def __init__(self, sim, nrn, syn, spiking=True) -> None:
        super().__init__(sim, nrn, syn)

        self.noise = None

        self.weight_scale = nrn["weight_scale"] if spiking else 1
        self.spiking = spiking
        self.use_mm = sim["use_mm"]  # use nest multimeter for recording neuron states
        if self.spiking:
            self.nrn["input"]["gamma"] = self.weight_scale
            self.nrn["pyr"]["gamma"] = self.weight_scale * nrn["pyr"]["gamma"]
            self.nrn["intn"]["gamma"] = self.weight_scale * nrn["intn"]["gamma"]
            for layer in range(len(self.dims)-1):
                for syn_name in ["ip", "up", "down", "pi"]:
                    if syn_name not in self.syn["conns"][layer]:
                        continue
                    synapse = self.syn["conns"][layer][syn_name]
                    if "eta" in synapse:
                        if syn_name == "pi":
                            synapse["eta"] /= self.weight_scale**2 * self.syn["tau_Delta"]
                        else:
                            synapse["eta"] /= self.weight_scale**3 * self.syn["tau_Delta"]
        self.setup_populations()

        self.iteration = 0

    def setup_populations(self):

        # Create input layer neurons
        self.input_neurons = nest.Create(self.nrn["model"], self.dims[0], self.nrn["input"])

        pyr_current = self.input_neurons
        intn_current = None
        eta = {}
        for i in range(len(self.dims)-2):
            conns_l = self.syn["conns"][i]
            eta["up"] = conns_l["up"]["eta"] if "eta" in conns_l["up"] else 0
            eta["pi"] = conns_l["pi"]["eta"] if "eta" in conns_l["pi"] else 0
            eta["ip"] = conns_l["ip"]["eta"] if "eta" in conns_l["ip"] else 0
            layer = NestLayer(deepcopy(self.nrn), deepcopy(self.sim),
                              deepcopy(self.syn), i, eta, pyr_current, intn_current)
            self.layers.append(layer)
            pyr_current = layer.pyr
            intn_current = layer.intn

        conns_l = self.syn["conns"][-1]
        eta["up"] = conns_l["up"]["eta"] if "eta" in conns_l["up"] else 0
        self.layers.append(NestOutputLayer(deepcopy(self.nrn), deepcopy(self.sim),
                                           deepcopy(self.syn), eta, pyr_current, intn_current))

        compartments = nest.GetDefaults(self.nrn["model"])["receptor_types"]
        foo = nest.GetConnections(self.layers[0].pyr, self.layers[0].intn)
        if self.sim["noise"]:
            # Inject Gaussian white noise into neuron somata.
            self.noise = nest.Create("noise_generator", 1, {"mean": 0., "std": self.sigma_noise})
            for l_current in self.layers[:-1]:
                nest.Connect(self.noise, l_current.pyr, syn_spec={"receptor_type": compartments["soma_curr"]})
                nest.Connect(self.noise, l_current.intn, syn_spec={"receptor_type": compartments["soma_curr"]})
            nest.Connect(self.noise, self.layers[-1].pyr, syn_spec={"receptor_type": compartments["soma_curr"]})

        if self.use_mm:
            self.mm = nest.Create('multimeter', 1, {'record_to': self.sim["recording_backend"],
                                                    'interval': self.sim["record_interval"],
                                                    'record_from': ["V_m.a_lat", "V_m.s", "V_m.b"]})
            nest.Connect(self.mm, self.input_neurons)
            nest.Connect(self.mm, self.layers[0].pyr)
            nest.Connect(self.mm, self.layers[0].intn)
            nest.Connect(self.mm, self.layers[-1].pyr)
        else:
            self.U_y_record = np.zeros((1, self.dims[-1]))
            self.V_ah_record = np.zeros((1, self.dims[1]))
            self.U_h_record = np.zeros((1, self.dims[1]))
            self.U_i_record = np.zeros((1, self.dims[-1]))

        # step generators for enabling batch training
        self.sgx = nest.Create("step_current_generator", self.dims[0])
        nest.Connect(self.sgx, self.input_neurons, conn_spec='one_to_one',
                     syn_spec={"receptor_type": compartments["soma_curr"]})
        self.sgy = nest.Create("step_current_generator", self.dims[-1])
        nest.Connect(self.sgy, self.layers[-1].pyr, conn_spec='one_to_one',
                     syn_spec={"receptor_type": compartments["soma_curr"]})

        # TODO: explain this shite
        pyr_prev = self.input_neurons
        for i in range(len(self.layers)-1):
            layer = self.layers[i]
            layer.redefine_connections(pyr_prev, self.layers[i+1].pyr)
            pyr_prev = layer.pyr
        self.layers[-1].redefine_connections(pyr_prev)

        for i in range(len(self.layers) - 1):
            l_current = self.layers[i]
            l_next = self.layers[i + 1]
            l_current.set_feedback_conns(l_next.pyr)
            if self.sim["init_self_pred"]:
                w_down = self.get_weight_array_from_syn(l_current.down)
                self.set_weights(-w_down, l_current.pi)
                w_up = self.get_weight_array_from_syn(l_next.up)

                self.set_weights(w_up * l_next.gb / (l_next.gl + l_next.ga + l_next.gb) *
                                 (l_current.gl + l_current.gd) / l_current.gd, l_current.ip)

    def simulate(self, T):
        # if self.sim["recording_backend"] == "ascii":
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
            self.input_neurons[i].set({"soma": {"I_e": input_currents[i] / self.nrn["tau_x"]}})

    def train_epoch(self, x_batch, y_batch):

        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            self.set_input(x)
            self.set_target(y)
            self.simulate(self.sim_time)
            if i == len(x)-1:
                U_y = [nrn.get("soma")["V_m"] for nrn in self.layers[-1].pyr]
                if not self.use_mm:
                    U_h = [nrn.get("soma")["V_m"] for nrn in self.layers[0].pyr]
                    V_ah = [nrn.get("apical_lat")["V_m"] for nrn in self.layers[0].pyr]
                    U_i = [nrn.get("soma")["V_m"] for nrn in self.layers[0].intn]

                    self.V_ah_record = np.concatenate((self.V_ah_record, np.expand_dims(V_ah, 0)), axis=0)
                    self.U_h_record = np.concatenate((self.U_h_record, np.expand_dims(U_h, 0)), axis=0)
                    self.U_i_record = np.concatenate((self.U_i_record, np.expand_dims(U_i, 0)), axis=0)
                    self.U_y_record = np.concatenate((self.U_y_record, np.expand_dims(U_y, 0)), axis=0)

                self.train_loss.append((self.epoch, mse(y, U_y)))
            self.reset()

    def test_teacher(self, n_samples=5):
        raise DeprecationWarning
        assert self.teacher
        loss = []
        for i in range(n_samples):
            x_test, y_actual = self.generate_teacher_data()

            WHX = self.get_weight_array(self.pyr_pops[0], self.pyr_pops[1])
            WYH = self.get_weight_array(self.pyr_pops[1], self.pyr_pops[2])
            y_pred = self.nrn["lambda_out"] * self.weight_scale * \
                WYH @ self.phi(self.nrn["lambda_ah"] * self.weight_scale * WHX @ x_test)
            loss.append(mse(y_actual, y_pred))
        self.test_loss.append(np.mean(loss))

    def test_bars(self, n_samples=8):
        acc = []
        loss_mse = []
        # set all learning rates to zero
        self.disable_learning()

        for sample_idx in range(n_samples):
            x_test, y_actual = self.generate_bar_data(sample_idx)
            self.set_input(x_test)
            self.simulate(self.sim_time)
            y_pred = [nrn.get("soma")["V_m"] for nrn in self.layers[-1].pyr]
            loss_mse.append(mse(y_actual, y_pred))
            acc.append(np.argmax(y_actual) == np.argmax(y_pred))
            self.reset()

        self.test_acc.append([self.epoch, np.mean(acc)])
        self.test_loss.append([self.epoch, np.mean(loss_mse)])

        # set learning rates to their original values
        self.enable_learning()

    def disable_learning(self):
        nest.GetConnections(synapse_model=self.syn["synapse_model"]).set({"eta": 0})

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
            self.layers[-1].pyr[i].set({"soma": {"I_e": target_currents[i] * self.nrn["g_som"]}})

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

    def set_weights(self, weights, synapse_collection):
        # TODO: match numpy variant
        for i, source_id in enumerate(sorted(set(synapse_collection.sources()))):
            for j, target_id in enumerate(sorted(set(synapse_collection.targets()))):
                source = nest.GetNodes({"global_id": source_id})
                target = nest.GetNodes({"global_id": target_id})
                nest.GetConnections(source, target).set({"weight": weights[j][i]})
