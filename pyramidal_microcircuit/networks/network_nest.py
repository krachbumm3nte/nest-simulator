import nest
import numpy as np
from .network import Network
from sklearn.metrics import mean_squared_error as mse
import pandas as pd


class NestNetwork(Network):

    def __init__(self, sim, nrn, syns) -> None:
        super().__init__(sim, nrn, syns)

        self.noise = None

        self.pyr_pops = []
        self.intn_pops = []
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
            synapse_yh = syns["hx"] if layer == 1 else syns["yh"]
            synapse_yh['weight'] = self.gen_weights(self.dims[layer-1], self.dims[layer])
            nest.Connect(pyr_pop_prev, pyr_pop, syn_spec=synapse_yh)

            if layer > 1:
                # Connect current to previous layer pyramidal populations
                synapse_hy = syns["hy"]
                synapse_hy['weight'] = self.gen_weights(
                    self.dims[layer], self.dims[layer-1])  # / (syns["wmax_init"] * nrn["gamma"])
                # TODO: perhaps we can get away with setting weights within the dict directly instead of creating variables here?
                nest.Connect(pyr_pop, pyr_pop_prev, syn_spec=synapse_hy)

                intn_pop = nest.Create(nrn["model"], self.dims[layer], nrn["intn"])
                self.intn_pops.append(intn_pop)
                # Set target IDs for pyr->intn current transmission
                for i in range(len(pyr_pop)):
                    pyr_pop[i].target = intn_pop[i].get("global_id")

                # Connect previous layer pyramidal to current layer interneuron populations
                synapse_ih = syns['ih']
                synapse_ih['weight'] = (synapse_yh['weight'] if self.sim["self_predicting_ff"]
                                        else self.gen_weights(self.dims[layer-1], self.dims[layer]))
                nest.Connect(pyr_pop_prev, intn_pop, syn_spec=synapse_ih)

                # Connect current layer interneuron to previous layer pyramidal populations
                synapse_hi = syns['hi']
                synapse_hi['weight'] = (-1 * synapse_hy['weight'] if self.sim["self_predicting_fb"]
                                        else self.gen_weights(self.dims[layer], self.dims[layer-1]))
                nest.Connect(intn_pop, pyr_pop_prev, syn_spec=synapse_hi)

        self.pyr_pops[-1].set({"apical_lat": {"g": 0}})

        if self.sim["noise"]:
            # Inject Gaussian white noise into neuron somata.
            self.noise = nest.Create("noise_generator", 1, {"mean": 0., "std": self.sigma_noise})
            compartments = nest.GetDefaults(nrn["model"])["receptor_types"]
            populations = self.pyr_pops[1:] + self.intn_pops  # Everything except input neurons receives noise.
            for pop in populations:
                nest.Connect(self.noise, pop, syn_spec={"receptor_type": compartments["soma_curr"]})

        # self.mm_x = nest.Create('multimeter', 1, {'record_to': self.sim["recording_backend"], 'record_from': ["V_m.s"]})
        # nest.Connect(self.mm_x, self.pyr_pops[0])
        self.mm = nest.Create(
            'multimeter', 1, {'record_to': self.sim["recording_backend"], 'record_from': ["V_m.a_lat", "V_m.s"]})
        nest.Connect(self.mm, self.pyr_pops[1])
        nest.Connect(self.mm, self.intn_pops[0])
        nest.Connect(self.mm, self.pyr_pops[2])

        # step generators for enabling batch training 
        self.sgx = nest.Create("step_current_generator")
        nest.Connect(self.sgx, self.pyr_pops[0], syn_spec={"receptor_type": compartments["soma_curr"]})        
        self.sgy = nest.Create("step_current_generator")
        nest.Connect(self.sgy, self.pyr_pops[-1], syn_spec={"receptor_type": compartments["soma_curr"]})


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
        for i in range(self.dims[0]):
            self.pyr_pops[0][i].set({"soma": {"I_e": input_currents[i] / self.nrn["tau_x"]}})

    def train(self, input_currents, T):

        self.set_input(input_currents)

        if self.teacher:
            self.calculate_target(input_currents)
            self.target_curr = self.phi_inverse(self.y)

            if not isinstance(self.target_curr, np.ndarray):
                self.target_curr = [self.target_curr]
            for i in range(self.dims[-1]):
                self.pyr_pops[-1].set({"soma": {"I_e": self.nrn["g_s"] * self.target_curr[i]}})

        self.simulate(T)

        if self.teacher:
            output_pred = [e["V_m"] for e in self.pyr_pops[-1].get("soma")]
            self.output_loss.append(mse(self.y, output_pred))

    def train_batches(self, T, batchsize):

        self.input_currents = np.random.random((batchsize, self.dims[0]))
        self.target_currents = np.zeros((batchsize, self.dims[-1]))

        t_now = nest.GetKernelStatus("biological_time")
        times = np.arange(t_now + self.sim["delta_t"], t_now + T * batchsize, T)



        for batch in range(batchsize):
            self.calculate_target(self.input_currents[batch])
            self.target_currents[batch,:] = self.phi_inverse(self.y)

        for i, step_generator in enumerate(self.sgx):
            step_generator.set(amplitude_values = self.input_currents[:,i]/self.nrn["tau_x"], amplitude_times = times)
        for i, step_generator in enumerate(self.sgy):
            step_generator.set(amplitude_values = self.target_currents[:,i], amplitude_times = times)

        self.simulate(T*batchsize)

        WHX = pd.DataFrame.from_dict(nest.GetConnections(source=self.pyr_pops[0], target=self.pyr_pops[1]).get())
        WYH = pd.DataFrame.from_dict(nest.GetConnections(source=self.pyr_pops[1], target=self.pyr_pops[2]).get())
        WHX = WHX.sort_values(["target", "source"]).weight.values.reshape((self.dims[1], self.dims[0]))
        WYH = WYH.sort_values(["target", "source"]).weight.values.reshape((self.dims[2], self.dims[1]))

        # calculte output loss for the last example of the batch
        y_pred = self.phi(self.nrn["lambda_out"] * np.matmul(WYH, self.phi(self.nrn["lambda_ah"] * np.matmul(WHX,  self.input_currents[-1,:]))))
        self.output_loss.append(mse(y_pred, np.asarray(self.y).flatten()))

    def set_target(self, target_currents):
        """Inject a constant current into all neurons in the output layer.

        @note: Before injection, currents are attenuated by the output neuron
        nudging conductance in order to match the simulation exactly.

        Arguments:
            input_currents -- Iterable of length equal to the output dimension.
        """
        # TODO: obsolete?
        for i in range(self.dims[-1]):
            self.pyr_pops[-1][i].set({"soma": {"I_e": self.phi_inverse(target_currents[i]) * self.nrn["g_s"]}})

    def get_weight_dict(self):
            
        weights = {}

        in_, hidden, out = self.pyr_pops
        interneurons = self.intn_pops[0]

        WHY = pd.DataFrame.from_dict(nest.GetConnections(source=out, target=hidden).get())
        WHI = pd.DataFrame.from_dict(nest.GetConnections(source=interneurons, target=hidden).get())
        WYH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=out).get())
        WIH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=interneurons).get())
        WHX = pd.DataFrame.from_dict(nest.GetConnections(source=in_, target=hidden).get())

        weights["hy"] = WHY.sort_values(["target", "source"]).weight.values.reshape((-1, len(out))).tolist()
        weights["hi"] = WHI.sort_values(["target", "source"]).weight.values.reshape((-1, len(interneurons))).tolist()
        weights["yh"] = WYH.sort_values(["target", "source"]).weight.values.reshape((-1, len(hidden))).tolist()
        weights["ih"] = WIH.sort_values(["target", "source"]).weight.values.reshape((-1, len(hidden))).tolist()
        weights["hx"] = WHX.sort_values(["target", "source"]).weight.values.reshape((-1, len(in_))).tolist()

        return weights