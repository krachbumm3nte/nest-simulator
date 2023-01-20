import nest
import numpy as np
from copy import deepcopy
from .network import Network


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
            pyr_pop_prev = self.pyr_pops[-1] # store previous layer pyramidal population

            pyr_pop = nest.Create(nrn["model"], self.dims[layer], nrn["pyr"])
            self.pyr_pops.append(pyr_pop)
            
            # Connect previous to current layer pyramidal populations
            synapse_yh = syns["hx"] if layer == 1 else syns["yh"]
            synapse_yh['weight'] = self.gen_weights(self.dims[layer-1], self.dims[layer])
            nest.Connect(pyr_pop_prev, pyr_pop, syn_spec=synapse_yh)

            if layer > 1:
                # Connect current to previous layer pyramidal populations
                synapse_hy = syns["hy"]
                synapse_hy['weight'] = self.gen_weights(self.dims[layer], self.dims[layer-1]) / (syns["wmax_init"] * nrn["gamma"])
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
            populations = self.pyr_pops[1:] + self.intn_pops # Everything except input neurons receives noise.
            for pop in populations:
                nest.Connect(self.noise, pop, syn_spec={"receptor_type": compartments["soma_curr"]})


        # self.mm_x = nest.Create('multimeter', 1, {'record_to': 'ascii', 'record_from': ["V_m.s"]})
        # nest.Connect(self.mm_x, self.pyr_pops[0])
        self.mm_h = nest.Create('multimeter', 1, {'record_to': 'ascii', 'record_from': ["V_m.a_lat", "V_m.s"]})
        nest.Connect(self.mm_h, self.pyr_pops[1])
        self.mm_i = nest.Create('multimeter', 1, {'record_to': 'ascii', 'record_from': ["V_m.s"]})
        nest.Connect(self.mm_i, self.intn_pops[0])
        self.mm_y = nest.Create('multimeter', 1, {'record_to': 'ascii', 'record_from': ["V_m.s"]})
        nest.Connect(self.mm_y, self.pyr_pops[2])

    def simulate(self, T):

        nest.SetKernelStatus({"data_prefix": f"it{self.iteration}_"})
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
        assert self.teacher

        self.set_input(input_currents)

        self.y = self.phi(self.yh_teacher * self.phi(self.hx_teacher * np.reshape(input_currents, (-1, 1))))
        self.y = self.phi_inverse(np.squeeze(np.asarray(self.y)))
        for i in range(self.dims[-1]):
            self.pyr_pops[-1].set({"soma": {"I_e": self.nrn["g_s"] * self.y[i]}})
        
        self.simulate(T)

        y_pred = self.phi(self.yh_teacher * self.phi(self.hx_teacher * np.reshape(input_currents, (-1, 1))))


    


        
        


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
