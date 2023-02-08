import nest
import numpy as np
from .network import Network
from sklearn.metrics import mean_squared_error as mse
import pandas as pd


class NestNetwork(Network):

    def __init__(self, sim, nrn, syn, spiking=True) -> None:
        super().__init__(sim, nrn, syn)

        self.noise = None

        self.pyr_pops = []
        self.intn_pops = []
        self.weight_scale = nrn["weight_scale"] if spiking else 1
        self.spiking = spiking
        if self.spiking:
            self.nrn["input"]["gamma"] = self.weight_scale
            self.nrn["pyr"]["gamma"] = self.weight_scale * nrn["pyr"]["gamma"]
            self.nrn["intn"]["gamma"] = self.weight_scale * nrn["intn"]["gamma"]
            self.syns['w_init_hx'] /= self.weight_scale
            self.syns['w_init_hi'] /= self.weight_scale
            self.syns['w_init_ih'] /= self.weight_scale
            self.syns['w_init_hy'] /= self.weight_scale
            self.syns['w_init_yh'] /= self.weight_scale
            for syn_name in ["hx", "yh", "hy", "ih"]:
                if "eta" in syn[syn_name]:
                    syn[syn_name]["eta"] /= self.weight_scale**3 * 30
            if "eta" in syn["hi"]:
                syn["hi"]["eta"] /= self.weight_scale**2 * 30

        self.setup_populations(self.syns, self.nrn)

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
                synapse_yh = syns["hx"]
                synapse_yh['weight'] = self.gen_weights(
                    self.dims[layer-1], self.dims[layer], -syns["w_init_hx"], syns["w_init_hx"])
            else:
                synapse_yh = syns["yh"]
                synapse_yh['weight'] = self.gen_weights(
                    self.dims[layer-1], self.dims[layer], -syns["w_init_yh"], syns["w_init_yh"])

            nest.Connect(pyr_pop_prev, pyr_pop, syn_spec=synapse_yh)

            if layer > 1:
                # Connect current to previous layer pyramidal populations

                syns["hy"]['weight'] = self.gen_weights(
                    self.dims[layer], self.dims[layer-1], -syns["w_init_hy"], syns["w_init_hy"])
                # TODO: perhaps we can get away with setting weights within the dict directly instead of creating variables here?
                nest.Connect(pyr_pop, pyr_pop_prev, syn_spec=syns["hy"])

                intn_pop = nest.Create(nrn["model"], self.dims[layer], nrn["intn"])
                self.intn_pops.append(intn_pop)
                # Set target IDs for pyr->intn current transmission
                for i in range(len(pyr_pop)):
                    pyr_pop[i].target = intn_pop[i].get("global_id")

                # Connect previous layer pyramidal to current layer interneuron populations
                syns['ih']['weight'] = (synapse_yh['weight'] if self.sim["self_predicting_ff"]
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
        self.mm = nest.Create(
            'multimeter', 1, {'record_to': self.sim["recording_backend"], 'record_from': ["V_m.a_lat", "V_m.s", "V_m.b"]})
        nest.Connect(self.mm, self.pyr_pops[0])
        nest.Connect(self.mm, self.pyr_pops[1])
        nest.Connect(self.mm, self.pyr_pops[2])
        nest.Connect(self.mm, self.intn_pops[0])

        # step generators for enabling batch training
        self.sgx = nest.Create("step_current_generator", self.dims[0])
        nest.Connect(self.sgx, self.pyr_pops[0], conn_spec='one_to_one', syn_spec={"receptor_type": compartments["soma_curr"]})
        self.sgy = nest.Create("step_current_generator", self.dims[-1])
        nest.Connect(self.sgy, self.pyr_pops[-1], conn_spec='one_to_one', syn_spec={"receptor_type": compartments["soma_curr"]})

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

    def train_batches_bars(self, batchsize):
        self.run_batch(batchsize, self.generate_bar_data)
        self.test_bars()


    def train_batches_teacher(self, batchsize):
        self.run_batch(batchsize, self.generate_teacher_data)
        self.test_teacher()

    def run_batch(self, batchsize, data_generator):
        x_batch = np.zeros((batchsize, self.dims[0]))
        y_batch = np.zeros((batchsize, self.dims[-1]))
        for i in range(batchsize):
            x, y = data_generator()
            x_batch[i] = x
            y_batch[i] = y

        t_now = nest.GetKernelStatus("biological_time")
        times = np.arange(t_now + self.sim["delta_t"], t_now + self.sim_time * batchsize, self.sim_time)

        for i, sg in enumerate(self.sgx):
            sg.set(amplitude_values=x_batch[:, i]/self.nrn["tau_x"], amplitude_times=times)
        for i, sg in enumerate(self.sgy):
            sg.set(amplitude_values=y_batch[:, i]*self.nrn["g_s"], amplitude_times=times)
        
        self.simulate(self.sim_time*batchsize)

        y_actual = y_batch[-1, :]
        y_pred = [nrn.get("soma")["V_m"] for nrn in self.pyr_pops[-1]]

        self.train_loss.append(mse(y_actual, y_pred))



    def test_teacher(self):
        assert self.teacher
        x_test, y_actual = self.generate_teacher_data()
        
        WHX = pd.DataFrame.from_dict(nest.GetConnections(source=self.pyr_pops[0], target=self.pyr_pops[1]).get())
        WYH = pd.DataFrame.from_dict(nest.GetConnections(source=self.pyr_pops[1], target=self.pyr_pops[2]).get())
        WHX = WHX.sort_values(["target", "source"]).weight.values.reshape((self.dims[1], self.dims[0]))
        WYH = WYH.sort_values(["target", "source"]).weight.values.reshape((self.dims[2], self.dims[1]))
        y_pred = self.nrn["lambda_out"] * self.weight_scale * \
            WYH @ self.phi(self.nrn["lambda_ah"] * self.weight_scale * WHX @ x_test)
        
        self.test_loss.append(mse(y_actual, y_pred))

    def test_bars(self):
        x_test, y_actual = self.generate_bar_data()

        WHX = pd.DataFrame.from_dict(nest.GetConnections(source=self.pyr_pops[0], target=self.pyr_pops[1]).get())
        WYH = pd.DataFrame.from_dict(nest.GetConnections(source=self.pyr_pops[1], target=self.pyr_pops[2]).get())
        WHX = WHX.sort_values(["target", "source"]).weight.values.reshape((self.dims[1], self.dims[0]))
        WYH = WYH.sort_values(["target", "source"]).weight.values.reshape((self.dims[2], self.dims[1]))
        y_pred = self.weight_scale * WYH @ self.phi(self.weight_scale * WHX @ x_test)
        
        self.test_loss.append(mse(y_actual, y_pred))

    def set_target(self, target_currents):
        """Inject a constant current into all neurons in the output layer.

        @note: Before injection, currents are attenuated by the output neuron
        nudging conductance in order to match the simulation exactly.

        Arguments:
            input_currents -- Iterable of length equal to the output dimension.
        """
        self.target_curr = target_currents
        for i in range(self.dims[-1]):
            self.pyr_pops[-1][i].set({"soma": {"I_e": target_currents[i] * self.nrn["g_s"]}})

    def get_weight_dict(self):

        weights = {}

        in_, hidden, out = self.pyr_pops
        interneurons = self.intn_pops[0]

        WHY = pd.DataFrame.from_dict(nest.GetConnections(source=out, target=hidden).get())
        WHI = pd.DataFrame.from_dict(nest.GetConnections(source=interneurons, target=hidden).get())
        WYH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=out).get())
        WIH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=interneurons).get())
        WHX = pd.DataFrame.from_dict(nest.GetConnections(source=in_, target=hidden).get())

        weights["hy"] = WHY.sort_values(["target", "source"]).weight.values.reshape((-1, len(out)))
        weights["hi"] = WHI.sort_values(["target", "source"]).weight.values.reshape((-1, len(interneurons)))
        weights["yh"] = WYH.sort_values(["target", "source"]).weight.values.reshape((-1, len(hidden)))
        weights["ih"] = WIH.sort_values(["target", "source"]).weight.values.reshape((-1, len(hidden)))
        weights["hx"] = WHX.sort_values(["target", "source"]).weight.values.reshape((-1, len(in_)))

        return weights
