import numpy as np
import pandas as pd
from src.networks.layer_NEST import NestLayer, NestOutputLayer
from src.networks.network import Network
from src.params import Params
from sklearn.metrics import mean_squared_error as mse
from copy import deepcopy
import nest


class NestNetwork(Network):

    def __init__(self, p: Params) -> None:
        super().__init__(p)

        self.noise_generator = None
        self.spiking = p.spiking
        self.weight_scale = self.p.weight_scale if self.spiking else 1
        self.recording_backend = "memory"         # backend for NEST multimeter recordings
        if self.p.record_interval <= 0:
            print("Disabling multimeter recroding.")
            self.use_mm = False  # flag to record activity of nest neurons using multimeters
        else:
            self.use_mm = True

        print("Setting up populations... ")
        self.setup_populations()
        print("Done")

    def setup_populations(self):
        self.wr = None
        if self.p.record_weights:
            # initialize a weight_recorder, and update all synapse models to interface with it
            self.wr = nest.Create("weight_recorder", params={'record_to': "ascii", "precision": 12})

            nest.CopyModel(self.p.syn_model, 'record_syn', {"weight_recorder": self.wr})
            self.p.syn_model = 'record_syn'

            nest.CopyModel(self.p.static_syn_model, 'static_record_syn', {"weight_recorder": self.wr})
            self.p.static_syn_model = 'static_record_syn'

        # set up dictionaries for neuron and synapse initialization
        self.p.setup_nest_configs()

        # Create input layer neurons
        # if self.spiking:
        #     self.poisson_generators = nest.Create("poisson_generator", self.dims[0])
        #     self.input_neurons = nest.Create("parrot_neuron", self.dims[0])
        #     nest.Connect(self.poisson_generators, self.input_neurons,
        #               conn_spec='one_to_one', syn_spec={'delay': self.p.delta_t})
        # else:
        #     # self.input_neurons = nest.Create("step_rate_generator", self.dims[0])
        self.input_neurons = nest.Create(self.p.neuron_model, self.dims[0], self.p.input_params)

        # Create hidden layers
        pyr_prev = self.input_neurons
        intn_prev = None
        for i in range(len(self.dims)-2):
            layer = NestLayer(self, self.p, i)
            self.layers.append(layer)
            pyr_prev = layer.pyr
            intn_prev = layer.intn

        # output layer
        self.layers.append(NestOutputLayer(self, self.p))

        self.output_stimulators = nest.Create(self.p.neuron_model, self.dims[-1], self.p.input_params)
        stim_synapse = deepcopy(self.p.syn_static)
        stim_synapse.update({"receptor_type": self.p.compartments["soma"],
                             "weight": self.p.g_som/self.weight_scale})

        nest.Connect(self.output_stimulators, self.layers[-1].pyr, conn_spec="one_to_one", syn_spec=stim_synapse)

        if self.p.noise:
            # Inject Gaussian white noise into neuron somata.
            self.noise_generator = nest.Create("noise_generator", 1, {"mean": 0., "std": self.sigma_noise})
            for layer in self.layers[:-1]:
                nest.Connect(self.noise_generator, layer.pyr, syn_spec={
                             "receptor_type": self.p.compartments["soma_curr"]})
                nest.Connect(self.noise_generator, layer.intn, syn_spec={
                             "receptor_type": self.p.compartments["soma_curr"]})
            nest.Connect(self.noise_generator, self.layers[-1].pyr,
                         syn_spec={"receptor_type": self.p.compartments["soma_curr"]})

        if self.use_mm:
            record_from = ["V_m.a_lat", "V_m.s", "V_m.b"] if self.p.store_errors else ["V_m.s"]
            self.mm = nest.Create('multimeter', 1, {'record_to': self.recording_backend,
                                                    'interval': self.p.record_interval,
                                                    'record_from': record_from,
                                                    'stop': 0.0  # disables multimeter by default
                                                    })
            nest.Connect(self.mm, self.layers[-1].pyr)
            if self.p.store_errors:
                nest.Connect(self.mm, self.layers[-2].pyr)
                nest.Connect(self.mm, self.layers[-2].intn)

        print("\tConnecting layers... ")
        pyr_prev = self.input_neurons
        intn_prev = None
        for i in range(len(self.dims)-2):
            layer = self.layers[i]
            pyr_next = self.layers[i+1].pyr

            # Initialize weights to the self-predicting state
            if self.p.init_self_pred:
                layer.synapses["pi"]["weight"] = -layer.synapses["down"]["weight"]

                if i > 0:
                    l_prev = self.layers[i-1]
                    layer.synapses["up"]["weight"] = l_prev.synapses["ip"]["weight"] * \
                        ((layer.gl + layer.ga + layer.gb) / layer.gb) * (l_prev.gd / (l_prev.gl + l_prev.gd))

            layer.connect(pyr_prev, pyr_next, intn_prev)
            pyr_prev = layer.pyr
            intn_prev = layer.intn
        self.layers[-1].connect(pyr_prev, intn_prev)

        if self.p.p_conn < 1.0:

            dropout = 1 - self.p.p_conn
            print(f"Processing neuron dropout of {np.round(100*dropout, 2)}% ...")

            all_neurons = nest.GetNodes(properties={"model": self.p.neuron_model})
            all_synapses = nest.GetConnections(source=all_neurons, target=all_neurons)

            n_synapses = len(all_synapses)
            conns_to_delete = np.random.choice(n_synapses, int(dropout * n_synapses), replace=False)

            print(f"{len(conns_to_delete)} synapses will be deleted... ", end="")

            for i in conns_to_delete:
                nest.Disconnect(all_synapses[i])

            print("Done.")

        pyr_prev = self.input_neurons
        for i in range(len(self.layers)-1):
            self.layers[i].redefine_connections(pyr_prev, self.layers[i+1].pyr)
            pyr_prev = self.layers[i].pyr
        self.layers[-1].redefine_connections(pyr_prev)

        # if self.p.init_self_pred:
        #     print("\tSetting self-predicting weight... ", end="")
        #     self.set_selfpredicting_weights()
        #     print("Done.")

    def set_selfpredicting_weights(self):
        """Initialize weights to the self-predicting state. Note that this approach is highly
        inefficient for large networks, as weights need to be set individually.
        """
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            l_next = self.layers[i + 1]
            w_down = self.get_weight_array_from_syn(layer.down)
            self.set_weights_from_syn(-w_down, layer.pi)

            w_up = self.get_weight_array_from_syn(l_next.up)
            self.set_weights_from_syn(w_up * l_next.gb / (l_next.gl + l_next.ga + l_next.gb) *
                                      (layer.gl + layer.gd) / layer.gd, layer.ip)

    def simulate(self, T, enable_recording=False, with_delay=True):
        if enable_recording and self.use_mm:
            self.mm.set({"start": self.p.out_lag if with_delay else 0,
                        'stop': self.t_pres, 'origin': nest.biological_time})
            if self.recording_backend == "ascii":
                nest.SetKernelStatus({"data_prefix": f"it{str(self.iteration).zfill(8)}_"})

        nest.Simulate(T)
        self.iteration += 1

    def disable_learning(self):
        nest.GetConnections(synapse_model=self.p.syn_model).set({"eta": 0})

    def enable_learning(self):
        for layer in self.layers:
            layer.enable_learning()

    def set_input(self, input_currents):
        """Inject a constant current into all neurons in the input layer.

        @note: Before injection, currents are attenuated by the input time constant
        in order to match synaptic filtering.

        Arguments:
            input_currents -- Iterable of length equal to the input dimension.
        """
        input_currents = np.array(input_currents)
        self.input_currents = input_currents
        for i in range(self.dims[0]):
            if self.spiking:
                # self.poisson_generators[i].rate = self.weight_scale * input_currents[i] * 1000
                self.input_neurons[i].set({"soma": {"I_e": input_currents[i] / self.tau_x}})

            else:
                self.input_neurons[i].set({"soma": {"I_e": input_currents[i] / self.tau_x}})
                # self.input_neurons[i].set({"amplitude_times": [nest.biological_time + self.dt],
                #                            "amplitude_values": [input_currents[i]]})

    def set_target(self, target_currents):
        """Inject a constant current into all neurons in the output layer.

        @note: Before injection, currents are attenuated by the output neuron
        nudging conductance in order to match the simulation exactly.

        Arguments:
            target_currents -- Iterable of length equal to the output dimension.
        """
        for i in range(self.dims[-1]):
            # self.layers[-1].pyr[i].set({"soma": {"I_e": target_currents[i] * self.p.g_som}})
            if self.spiking:
                self.output_stimulators[i].set({"soma": {"I_e": 10 * target_currents[i] / self.tau_x}})
            else:
                self.output_stimulators[i].set({"soma": {"I_e":  target_currents[i] / self.tau_x}})

    def train_batch(self, x_batch, y_batch):
        loss = []
        for x, y in zip(x_batch, y_batch):
            self.reset()
            self.set_input(x)
            self.set_target(y)
            self.simulate(self.t_pres, enable_recording=True)
            if self.use_mm:
                mm_data = pd.DataFrame.from_dict(self.mm.events)
                y_pred = [mm_data[mm_data["senders"] == out_id]["V_m.s"] for out_id in self.layers[-1].pyr.global_id]
                y_pred = np.mean(y_pred, axis=1)
            else:
                y_pred = [e["V_m"] for e in self.layers[-1].pyr.get("soma")]
            loss.append(mse(y_pred, y))

        if self.p.store_errors and self.use_mm:
            U_I = [mm_data[mm_data["senders"] == intn_id]["V_m.s"] for intn_id in self.layers[-2].intn.global_id]
            U_I = np.mean(U_I, axis=1)
            V_ah = [mm_data[mm_data["senders"] == hidden_id]["V_m.a_lat"]
                    for hidden_id in self.layers[-2].pyr.global_id]
            V_ah = np.mean(V_ah, axis=1)
            # self.apical_error.append((self.epoch, float(np.linalg.norm(V_ah))))
            self.apical_error.append((self.epoch, float(np.abs(np.mean(V_ah)))))
            self.intn_error.append([self.epoch, mse(self.phi(U_I), self.phi(y_pred))])

        return np.mean(loss)

    def test_batch(self, x_batch, y_batch):
        acc = []
        loss_mse = []
        # set all learning rates to zero

        self.disable_learning()

        for x_test, y_actual in zip(x_batch, y_batch):
            self.set_input(x_test)
            if self.use_mm:
                # self.mm.set({"start": self.p.out_lag, 'stop': self.t_pres, 'origin': nest.biological_time})
                self.mm.set({"start": self.p.test_delay, 'stop': self.p.test_time, 'origin': nest.biological_time})
            self.simulate(self.p.test_time)  # self.t_pres)
            if self.use_mm:
                mm_data = pd.DataFrame.from_dict(self.mm.events)
                U_Y = [mm_data[mm_data["senders"] == out_id]["V_m.s"] for out_id in self.layers[-1].pyr.global_id]
                y_pred = np.mean(U_Y, axis=1)
            else:
                y_pred = [e["V_m"] for e in self.layers[-1].pyr.get("soma")]
            loss_mse.append(mse(y_actual, y_pred))
            acc.append(np.argmax(y_actual) == np.argmax(y_pred))
            self.reset()

        # set learning rates to their original values
        self.enable_learning()
        return np.mean(acc), np.mean(loss_mse)

    def get_weight_array(self, source, target, normalized=False):
        weight_df = pd.DataFrame.from_dict(nest.GetConnections(source=source, target=target).get())
        n_out = len(target)
        n_in = len(source)
        if self.p.p_conn == 1:
            weight_array = weight_df.sort_values(["target", "source"]).weight.values.reshape((n_out, n_in))
        else:
            weight_array = np.full((n_out, n_in), np.nan)
            for idx, w in weight_df.iterrows():
                weight_array[w["target"] % n_out, w["source"] % n_in] = w["weight"]

        if normalized:
            weight_array *= self.weight_scale
        return weight_array

    def get_weight_array_from_syn(self, synapse_collection, normalized=False):
        weight_df = pd.DataFrame.from_dict(synapse_collection.get())
        n_out = len(set(synapse_collection.targets()))
        n_in = len(set(synapse_collection.sources()))
        if self.p.p_conn == 1:
            weight_array = weight_df.sort_values(["target", "source"]).weight.values.reshape((n_out, n_in))
        else:
            weight_array = np.full((n_out, n_in), np.nan)
            for idx, w in weight_df.iterrows():
                weight_array[w["target"] % n_out, w["source"] % n_in] = w["weight"]

        if normalized:
            weight_array *= self.weight_scale
        return weight_array

    def get_weight_dict(self, normalized=True):
        weights = []
        pyr_prev = self.input_neurons
        for i, layer in enumerate(self.layers[:-1]):
            weights.append({"up": self.get_weight_array(pyr_prev, layer.pyr, normalized),
                            "pi": self.get_weight_array(layer.intn, layer.pyr, normalized),
                            "ip": self.get_weight_array(layer.pyr, layer.intn, normalized),
                            "down": self.get_weight_array(self.layers[i+1].pyr, layer.pyr, normalized)})
            pyr_prev = layer.pyr
        weights.append({"up": self.get_weight_array(pyr_prev, self.layers[-1].pyr, normalized)})
        return weights

    def reset(self):
        self.set_target(np.zeros(self.dims[-1]))
        self.set_input(np.zeros(self.dims[0]))
        if self.p.reset == 2:
            # hard reset
            for layer in self.layers:
                layer.reset()
        elif self.p.reset == 1:
            # soft reset
            for i in range(5):
                nest.Simulate(5)
        if self.use_mm:
            self.mm.n_events = 0

    def set_weights_from_syn(self, weights, synapse_collection):
        for i, source_id in enumerate(sorted(set(synapse_collection.sources()))):
            for j, target_id in enumerate(sorted(set(synapse_collection.targets()))):
                source = nest.GetNodes({"global_id": source_id})
                target = nest.GetNodes({"global_id": target_id})
                nest.GetConnections(source, target).set({"weight": weights[j][i]})

    def set_all_weights(self, weight_dict, normalized=True):
        print("setting all network weights... ", end="")
        if normalized:
            for i, layer in enumerate(weight_dict):
                for k, v in layer.items():
                    weight_dict[i][k] = np.asarray(v) / self.weight_scale
        for i, layer in enumerate(self.layers[:-1]):
            self.set_weights_from_syn(weight_dict[i]["up"], layer.up)
            self.set_weights_from_syn(weight_dict[i]["ip"], layer.ip)
            self.set_weights_from_syn(weight_dict[i]["pi"], layer.pi)
            self.set_weights_from_syn(weight_dict[i]["down"], layer.down)
        self.set_weights_from_syn(weight_dict[-1]["up"], self.layers[-1].up)
        print("Done")
