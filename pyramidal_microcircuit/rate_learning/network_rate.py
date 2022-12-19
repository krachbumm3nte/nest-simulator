import nest
from params_rate import *
import numpy as np


class Network:

    def __init__(self, dims) -> None:
        self.dims = dims
        self.noise = noise
        self.noise_std = noise_std
        self.stim_amp = stim_amp
        self.target_amp = target_amp
        self.L = len(dims)
        self.nudging = nudging
        self.pyr_pops = []
        self.intn_pops = []
        self.parrots = None
        self.gauss = None
        self.setup_populations(init_self_pred)

    def gen_weights(self, lr, next_lr, w_min=wmin_init, w_max=wmax_init):
        return np.random.uniform(w_min, w_max, (next_lr, lr))

    def setup_populations(self, self_predicting):

        self.pyr_pops.append(nest.Create(pyr_model, self.dims[0], input_params))

        if self.noise:
            gauss = nest.Create("noise_generator", 1, {"mean": 0., "std": self.noise_std})

        for l in range(1, self.L):
            pyr_l = nest.Create(pyr_model, self.dims[l], pyr_params)

            syn_spec_ff_pp = syn_yh
            syn_spec_ff_pp['weight'] = self.gen_weights(self.dims[l-1], self.dims[l])
            syn_spec_ff_pp['eta'] = eta_hx if l == 1 else eta_yh
            pyr_prev = self.pyr_pops[-1]
            if self.noise:
                nest.Connect(gauss, pyr_l, syn_spec={"receptor_type": pyr_comps["soma_curr"]})
            self.pyr_pops.append(pyr_l)
            nest.Connect(pyr_prev, pyr_l, syn_spec=syn_spec_ff_pp)

            if l > 1:
                syn_spec_fb_pp = syn_hy
                syn_spec_fb_pp['weight'] = self.gen_weights(self.dims[l], self.dims[l-1])
                nest.Connect(pyr_l, pyr_prev, syn_spec=syn_spec_fb_pp)

                int_l = nest.Create(intn_model, self.dims[l], intn_params)
                if self.noise:
                    nest.Connect(gauss, int_l, syn_spec={"receptor_type": intn_comps["soma_curr"]})

                if self.nudging:
                    for i in range(len(pyr_l)):
                        id = int_l[i].get("global_id")
                        pyr_l[i].target = id

                syn_spec_ff_pi = syn_ih
                syn_spec_ff_pi['weight'] = (
                    syn_spec_ff_pp['weight'] if self_predicting_ff else self.gen_weights(self.dims[l-1], self.dims[l]))
                nest.Connect(pyr_prev, int_l, syn_spec=syn_spec_ff_pi)

                syn_spec_fb_pi = syn_hi
                syn_spec_fb_pi['weight'] = (-1 * syn_spec_fb_pp['weight']
                                            if self_predicting_fb else self.gen_weights(self.dims[l], self.dims[l-1]))
                nest.Connect(int_l, pyr_prev, syn_spec=syn_spec_fb_pi)

                self.intn_pops.append(int_l)

        # Set special parameters for some of the populations:

        # output neurons are modeled without an apical compartment
        self.pyr_pops[-1].set({"apical_lat": {"g": 0}, })

        self.mm_x = nest.Create('multimeter', 1, {'record_from': ["V_m.s"]})
        self.mm_h = nest.Create('multimeter', 1, {'record_from': ["V_m.a_lat", "V_m.s", "V_m.b"]})
        self.mm_i = nest.Create('multimeter', 1, {'record_from': ["V_m.s", "V_m.b"]})
        self.mm_y = nest.Create('multimeter', 1, {'record_from': ["V_m.s", "V_m.b"]})

        nest.Connect(self.mm_x, self.pyr_pops[0])
        nest.Connect(self.mm_h, self.pyr_pops[1])
        nest.Connect(self.mm_i, self.intn_pops[0])
        nest.Connect(self.mm_y, self.pyr_pops[2])

        # self.sr_in = nest.Create("spike_recorder", 1)
        # self.sr_intn = nest.Create("spike_recorder", 1)
        # self.sr_hidden = nest.Create("spike_recorder", 1)
        # self.sr_out = nest.Create("spike_recorder", 1)
        # nest.Connect(self.pyr_pops[0], self.sr_in)
        # nest.Connect(self.intn_pops[0], self.sr_intn)
        # nest.Connect(self.pyr_pops[1], self.sr_hidden)
        # nest.Connect(self.pyr_pops[2], self.sr_out)

    def set_input(self, input_currents):
        """Inject a constant current into all neurons in the input layer.

        @note: Before injection, currents are attenuated by the input time constant
        in order to match the simulation exactly.

        Arguments:
            input_currents -- Iterable of length equal to the input dimension.
        """
        for i in range(self.dims[0]):
            self.pyr_pops[0][i].set({"soma": {"I_e": input_currents[i] * tau_input}})

    def set_target(self, indices):
        """Inject a constant current into all neurons in the output layer.

        @note: Before injection, currents are attenuated by the output neuron
        nudging conductance in order to match the simulation exactly.

        Arguments:
            indices -- Iterable of length equal to the output dimension.
        """
        for i in range(self.dims[-1]):
            if i in indices:
                self.pyr_pops[-1][i].set({"soma": {"I_e": self.stim_amp * g_s}})
            else:
                self.pyr_pops[-1][i].set({"soma": {"I_e": 0}})
