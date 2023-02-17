import nest
import numpy as np
from .network import Network
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from copy import deepcopy


class NestNetwork(Network):

    def __init__(self, sim, nrn, syn, spiking=True) -> None:
        super().__init__(sim, nrn, syn)

        self.noise = None

        self.pyr_pops = []
        self.intn_pops = []
        self.weight_scale = nrn["weight_scale"] if spiking else 1
        self.spiking = spiking
        self.use_mm = sim["use_mm"]  # use nest multimeter for recording neuron states
        if self.spiking:
            self.nrn["input"]["gamma"] = self.weight_scale
            self.nrn["pyr"]["gamma"] = self.weight_scale * nrn["pyr"]["gamma"]
            self.nrn["intn"]["gamma"] = self.weight_scale * nrn["intn"]["gamma"]
            self.syn['w_init_hx'] /= self.weight_scale
            self.syn['w_init_hi'] /= self.weight_scale
            self.syn['w_init_ih'] /= self.weight_scale
            self.syn['w_init_hy'] /= self.weight_scale
            self.syn['w_init_yh'] /= self.weight_scale
            for syn_name in ["hx", "yh", "hy", "ih"]:
                if "eta" in self.syn[syn_name]:
                    self.syn[syn_name]["Wmin"] /= self.weight_scale 
                    self.syn[syn_name]["Wmax"] /= self.weight_scale 
                    self.syn[syn_name]["eta"] /= self.weight_scale**3 * self.syn["tau_Delta"]
            if "eta" in syn["hi"]:
                self.syn["hi"]["Wmin"] /= self.weight_scale 
                self.syn["hi"]["Wmax"] /= self.weight_scale 
                self.syn["hi"]["eta"] /= self.weight_scale**2 * self.syn["tau_Delta"]

        self.setup_populations(deepcopy(self.syn), self.nrn)

    def setup_populations(self, syns, nrn):

        # Create input layer neurons
        self.pyr_pops.append(nest.Create(nrn["model"], self.dims[0], nrn["input"]))

        # Create and connect all subsequent populations
        for layer in range(1, len(self.dims)):
            pyr_pop_prev = self.pyr_pops[-1]  # store previous layer pyramidal population

            pyr_pop = nest.Create(nrn["model"], self.dims[layer], nrn["pyr"])
            self.pyr_pops.append(pyr_pop)

            # Connect previous to current layer pyramidal populations
            if layer == 1:
                synapse_ff_pyr = deepcopy(syns["hx"])
                synapse_ff_pyr['weight'] = self.gen_weights(
                    self.dims[layer-1], self.dims[layer], -syns["w_init_hx"], syns["w_init_hx"])
            else:
                synapse_ff_pyr = deepcopy(syns["yh"])
                synapse_ff_pyr['weight'] = self.gen_weights(
                    self.dims[layer-1], self.dims[layer], -syns["w_init_yh"], syns["w_init_yh"])

            nest.Connect(pyr_pop_prev, pyr_pop, syn_spec=synapse_ff_pyr)

            if layer > 1:
                # Connect current to previous layer pyramidal populations

                synapse_hy = syns["hy"]
                synapse_hy['weight'] = self.gen_weights(
                    self.dims[layer], self.dims[layer-1], -syns["w_init_hy"], syns["w_init_hy"])
                # TODO: perhaps we can get away with setting weights within the dict directly instead of creating variables here?
                nest.Connect(pyr_pop, pyr_pop_prev, syn_spec=syns["hy"])

                intn_pop = nest.Create(nrn["model"], self.dims[layer], nrn["intn"])
                self.intn_pops.append(intn_pop)
                # Set target IDs for pyr->intn current transmission
                for i in range(len(pyr_pop)):
                    pyr_pop[i].target = intn_pop[i].get("global_id")

                # Connect previous layer pyramidal to current layer interneuron populations
                syns['ih']['weight'] = (synapse_ff_pyr['weight'] if self.sim["self_predicting_ff"]
                                        else self.gen_weights(self.dims[layer-1], self.dims[layer], -syns["w_init_ih"], syns["w_init_ih"]))
                nest.Connect(pyr_pop_prev, intn_pop, syn_spec=syns['ih'])

                # Connect current layer interneuron to previous layer pyramidal populations
                syns['hi']['weight'] = (-1 * syns['hy']['weight'] if self.sim["self_predicting_fb"]
                                        else self.gen_weights(self.dims[layer], self.dims[layer-1], -syns["w_init_hi"], syns["w_init_hi"]))
                nest.Connect(intn_pop, pyr_pop_prev, syn_spec=syns['hi'])

        self.pyr_pops[-1].set({"apical_lat": {"g": 0}})
        compartments = nest.GetDefaults(nrn["model"])["receptor_types"]

        if self.sim["noise"]:
            # Inject Gaussian white noise into neuron somata.
            self.noise = nest.Create("noise_generator", 1, {"mean": 0., "std": self.sigma_noise})
            populations = self.pyr_pops[1:] + self.intn_pops  # Everything except input neurons receives noise.
            for pop in populations:
                nest.Connect(self.noise, pop, syn_spec={"receptor_type": compartments["soma_curr"]})

        # self.mm_x = nest.Create('multimeter', 1, {'record_to': self.sim["recording_backend"], 'record_from': ["V_m.s"]})
        # nest.Connect(self.mm_x, self.pyr_pops[0])
        if self.use_mm:
            self.mm = nest.Create(
                'multimeter', 1, {'record_to': self.sim["recording_backend"], 'record_from': ["V_m.a_lat", "V_m.s", "V_m.b"]})
            nest.Connect(self.mm, self.pyr_pops[0])
            nest.Connect(self.mm, self.pyr_pops[1])
            nest.Connect(self.mm, self.pyr_pops[2])
            nest.Connect(self.mm, self.intn_pops[0])
        else:
            self.U_y_record = np.zeros((1, self.dims[-1]))
            self.V_ah_record = np.zeros((1, self.dims[1]))
            self.U_h_record = np.zeros((1, self.dims[1]))
            self.U_i_record = np.zeros((1, self.dims[-1]))
        
        #TODO: only works for 3 layers!
        self.in_, self.hidden, self.out = self.pyr_pops
        self.interneurons = self.intn_pops[0]

        self.hy = nest.GetConnections(source=self.out, target=self.hidden)
        self.hi = nest.GetConnections(source=self.interneurons, target=self.hidden)
        self.yh = nest.GetConnections(source=self.hidden, target=self.out)
        self.ih = nest.GetConnections(source=self.hidden, target=self.interneurons)
        self.hx = nest.GetConnections(source=self.in_, target=self.hidden)
        self.conns = [self.hx, self.yh, self.ih, self.hi, self.hy]

        # step generators for enabling batch training
        self.sgx = nest.Create("step_current_generator", self.dims[0])
        nest.Connect(self.sgx, self.pyr_pops[0], conn_spec='one_to_one',
                     syn_spec={"receptor_type": compartments["soma_curr"]})
        self.sgy = nest.Create("step_current_generator", self.dims[-1])
        nest.Connect(self.sgy, self.pyr_pops[-1], conn_spec='one_to_one',
                     syn_spec={"receptor_type": compartments["soma_curr"]})

    def simulate(self, T):
        # if self.sim["recording_backend"] == "ascii":
        nest.SetKernelStatus({"data_prefix": f"it{str(self.iteration).zfill(8)}_"})
        nest.Simulate(T)

        self.iteration += 1

    def set_input(self, input_currents):
        """Inject a constant current into all neurons in the input layer.

        @note: Before injection, currents are attenuated by the input time constant
        in order to match the simulation exactly.

        Arguments:
            input_currents -- Iterable of length equal to the input dimension.
        """
        self.input_currents = input_currents
        for i in range(self.dims[0]):
            self.pyr_pops[0][i].set({"soma": {"I_e": input_currents[i] / self.nrn["tau_x"]}})

    def train_epoch(self, x_batch, y_batch):

        # l_epoch = len(x_batch)

        # t_now = nest.GetKernelStatus("biological_time")
        # times = np.arange(t_now + self.sim["delta_t"], t_now + self.sim_time * l_epoch, self.sim_time)
        # for i, sg in enumerate(self.sgx):
        #     sg.set(amplitude_values=x_batch[:, i]/self.nrn["tau_x"], amplitude_times=times)
        # for i, sg in enumerate(self.sgy):
        #     sg.set(amplitude_values=y_batch[:, i]*self.nrn["g_som"], amplitude_times=times)

        # self.simulate(self.sim_time*l_epoch)

        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            self.set_input(x)
            self.set_target(y)
            self.simulate(self.sim_time)

            if i == len(x)-1:
                U_y = [nrn.get("soma")["V_m"] for nrn in self.pyr_pops[-1]]
                if not self.use_mm:
                    U_h = [nrn.get("soma")["V_m"] for nrn in self.pyr_pops[1]]
                    V_ah = [nrn.get("apical_lat")["V_m"] for nrn in self.pyr_pops[1]]
                    U_i = [nrn.get("soma")["V_m"] for nrn in self.intn_pops[0]]

                    self.V_ah_record = np.concatenate((self.V_ah_record, np.expand_dims(V_ah, 0)), axis=0)
                    self.U_h_record = np.concatenate((self.U_h_record, np.expand_dims(U_h, 0)), axis=0)
                    self.U_i_record = np.concatenate((self.U_i_record, np.expand_dims(U_i, 0)), axis=0)
                    self.U_y_record = np.concatenate((self.U_y_record, np.expand_dims(U_y, 0)), axis=0)

                self.train_loss.append(mse(y, U_y))
            self.reset()

    def test_teacher(self, n_samples=5):
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

        # grab all connections with plastic synapses and set learning rate to 0
        nest.GetConnections(synapse_model=self.syn["synapse_model"]).set({"eta": 0})

        for i in range(n_samples):
            x_test, y_actual = self.generate_bar_data(i)
            # t_now = nest.GetKernelStatus("biological_time") + self.delta_t
            # for i, sg in enumerate(self.sgx):
            #     sg.set(amplitude_values=[x_test[i]/self.nrn["tau_x"]], amplitude_times=[t_now])
            # for i, sg in enumerate(self.sgy):
            #     sg.set(amplitude_values=[0], amplitude_times=[t_now])
            self.set_input(x_test)
            self.simulate(self.sim_time)
            y_pred = [nrn.get("soma")["V_m"] for nrn in self.pyr_pops[-1]]
            loss_mse.append(mse(y_actual, y_pred))
            acc.append(np.argmax(y_actual) == np.argmax(y_pred))

            self.reset()
        self.test_acc.append(np.mean(acc))
        self.test_loss.append(np.mean(loss_mse))

        # set learning rates to their original values
        nest.GetConnections(self.pyr_pops[0], self.pyr_pops[1]).set({"eta": self.syn["hx"]["eta"]})
        nest.GetConnections(self.pyr_pops[1], self.intn_pops[0]).set({"eta": self.syn["ih"]["eta"]})
        hi = nest.GetConnections(self.intn_pops[0], self.pyr_pops[1])
        if "eta" in hi.get():
            hi.set({"eta": self.syn["hi"]["eta"]})
        nest.GetConnections(self.pyr_pops[1], self.pyr_pops[2]).set({"eta": self.syn["yh"]["eta"]})

    def set_target(self, target_currents):
        """Inject a constant current into all neurons in the output layer.

        @note: Before injection, currents are attenuated by the output neuron
        nudging conductance in order to match the simulation exactly.

        Arguments:
            input_currents -- Iterable of length equal to the output dimension.
        """
        self.target_curr = target_currents
        for i in range(self.dims[-1]):
            self.pyr_pops[-1][i].set({"soma": {"I_e": target_currents[i] * self.nrn["g_som"]}})

    def get_weight_array(self, source, target):
        weights = pd.DataFrame.from_dict(nest.GetConnections(source=source, target=target).get())
        return weights.sort_values(["target", "source"]).weight.values.reshape((len(target), len(source))) * self.weight_scale

    def get_weight_dict(self):
        weights = {}
        weights["hy"] = self.get_weight_array(self.out, self.hidden)
        weights["hi"] = self.get_weight_array(self.interneurons,self.hidden)
        weights["yh"] = self.get_weight_array(self.hidden, self.out)
        weights["ih"] = self.get_weight_array(self.hidden, self.interneurons)
        weights["hx"] = self.get_weight_array(self.in_, self.hidden)

        return weights

    def reset(self):
        for pop in self.pyr_pops + self.intn_pops:
            pop.set({"soma": {"V_m": 0, "I_e": 0}, "basal": {"V_m": 0, "I_e": 0}, "apical_lat": {"V_m": 0, "I_e": 0}})

    def set_weights(self, weights):
        self.conn_names = ["hx", "yh", "ih", "hi", "hy"]

        for name, (targets, sources) in zip(self.conn_names, [[self.hidden, self.in_], [self.out, self.hidden], [self.interneurons, self.hidden], [self.hidden, self.interneurons], [self.hidden, self.out]]):
            for i, source in enumerate(sources):
                for j, target in enumerate(targets):
                    nest.GetConnections(source, target).set({"weight": weights[name][j][i]/ self.weight_scale})
