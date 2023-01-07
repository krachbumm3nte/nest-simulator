import nest
import numpy as np


class Network:

    def __init__(self, sim, nrn, syns) -> None:
        self.sim = sim  # simulation parameters
        self.nrn = nrn  # neuron parameters
        self.syns = syns  # synapse parameters

        self.dims = sim["dims"]
        self.pyr_pops = []
        self.intn_pops = []
        self.noise = None
        self.sigma_noise = sim["sigma"]
        self.phi = nrn["phi"]
        self.phi_inverse = nrn["phi_inverse"]
        self.setup_populations(syns, nrn)

        self.iteration = 0

    def gen_weights(self, lr, next_lr):
        return np.random.uniform(self.nrn["wmin_init"], self.nrn["wmax_init"], (next_lr, lr))

    def setup_populations(self, syns, nrn):

        comps = nest.GetDefaults(nrn["model"])["receptor_types"]

        self.pyr_pops.append(nest.Create(nrn["model"], self.dims[0], nrn["input"]))

        if self.noise:
            self.noise = nest.Create("noise_generator", 1, {"mean": 0., "std": self.sigma_noise})

        for layer in range(1, len(self.dims)):
            pyr_l = nest.Create(nrn["model"], self.dims[layer], nrn["pyr"])

            syn_spec_ff_pp = syns["hx"] if layer == 1 else syns["yh"]
            syn_spec_ff_pp['weight'] = self.gen_weights(self.dims[layer-1], self.dims[layer])

            pyr_prev = self.pyr_pops[-1]
            if self.noise:
                nest.Connect(self.noise, pyr_l, syn_spec={"receptor_type": comps["soma_curr"]})
            self.pyr_pops.append(pyr_l)
            nest.Connect(pyr_prev, pyr_l, syn_spec=syn_spec_ff_pp)

            if layer > 1:
                syn_spec_fb_pp = syns["hy"]
                syn_spec_fb_pp['weight'] = self.gen_weights(self.dims[layer], self.dims[layer-1])
                nest.Connect(pyr_l, pyr_prev, syn_spec=syn_spec_fb_pp)

                int_l = nest.Create(nrn["model"], self.dims[layer], nrn["intn"])
                if self.noise:
                    nest.Connect(self.noise, int_l, syn_spec={"receptor_type": comps["soma_curr"]})

                for i in range(len(pyr_l)):
                    id = int_l[i].get("global_id")
                    pyr_l[i].target = id

                syn_spec_ff_pi = syns['ih']
                syn_spec_ff_pi['weight'] = (syn_spec_ff_pp['weight'] if self.sim["self_predicting_ff"]
                                            else self.gen_weights(self.dims[layer-1], self.dims[layer]))
                nest.Connect(pyr_prev, int_l, syn_spec=syn_spec_ff_pi)

                syn_spec_fb_pi = syns['hi']
                syn_spec_fb_pi['weight'] = (-1 * syn_spec_fb_pp['weight'] if self.sim["self_predicting_fb"]
                                            else self.gen_weights(self.dims[layer], self.dims[layer-1]))
                nest.Connect(int_l, pyr_prev, syn_spec=syn_spec_fb_pi)

                self.intn_pops.append(int_l)

        # self.mm_x = nest.Create('multimeter', 1, {'record_to': 'ascii', 'record_from': ["V_m.s"]})
        self.mm_h = nest.Create('multimeter', 1, {'record_to': 'ascii', 'record_from': ["V_m.a_lat", "V_m.s"]})
        self.mm_i = nest.Create('multimeter', 1, {'record_to': 'ascii', 'record_from': ["V_m.s"]})
        self.mm_y = nest.Create('multimeter', 1, {'record_to': 'ascii', 'record_from': ["V_m.s"]})

        # nest.Connect(self.mm_x, self.pyr_pops[0])
        nest.Connect(self.mm_h, self.pyr_pops[1])
        nest.Connect(self.mm_i, self.intn_pops[0])
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

    def set_target(self, target_currents):
        """Inject a constant current into all neurons in the output layer.

        @note: Before injection, currents are attenuated by the output neuron
        nudging conductance in order to match the simulation exactly.

        Arguments:
            input_currents -- Iterable of length equal to the output dimension.
        """
        for i in range(self.dims[-1]):
            self.pyr_pops[-1][i].set({"soma": {"I_e": self.phi_inverse(target_currents[i]) * self.nrn["g_s"]}})
