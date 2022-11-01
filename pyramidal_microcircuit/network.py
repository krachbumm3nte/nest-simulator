import nest
from params import *
import numpy as np


class Network:

    def __init__(self, dims, noise=True, noise_std=0.1, init_self_pred=False, nudging=True) -> None:
        self.dims = dims
        self.noise = noise
        self.noise_std = noise_std
        self.stim_amp = 15
        self.target_amp = 0.5
        self.L = len(dims)
        self.nudging = nudging
        self.pyr_pops = []
        self.intn_pops = []
        self.parrots = None
        self.gauss = None
        self.setup_populations(init_self_pred)

    def gen_weights(self, lr, next_lr, w_min=-1, w_max=1):
        return np.random.uniform(w_min, w_max, (next_lr, lr))

    def setup_populations(self, self_predicting):

        self.pyr_pops.append(nest.Create(pyr_model, self.dims[0], pyr_params))

        if self.noise:
            gauss = nest.Create("noise_generator", 1, {"mean": 0., "std": self.noise_std})

        for lr in range(1, self.L):
            pyr_l = nest.Create(pyr_model, self.dims[lr], pyr_params)

            syn_spec_ff_pp = syn_ff_pyr_pyr
            syn_spec_ff_pp['weight'] = self.gen_weights(self.dims[lr-1], self.dims[lr])

            pyr_prev = self.pyr_pops[-1]
            if self.noise:
                nest.Connect(gauss, pyr_l, syn_spec={"receptor_type": pyr_comps["soma_curr"]})
            self.pyr_pops.append(pyr_l)
            nest.Connect(pyr_prev, pyr_l, syn_spec=syn_spec_ff_pp)

            if lr > 1:
                syn_spec_fb_pp = syn_fb_pyr_pyr
                syn_spec_fb_pp['weight'] = self.gen_weights(self.dims[lr], self.dims[lr-1])
                nest.Connect(pyr_l, pyr_prev, syn_spec=syn_spec_fb_pp)

                int_l = nest.Create(intn_model, self.dims[lr], intn_params)
                if self.noise:
                    nest.Connect(gauss, int_l, syn_spec={"receptor_type": intn_comps["soma_curr"]})

                if self.nudging:
                    for i in range(len(pyr_l)):
                        print("setting target")
                        id = int_l[i].get("global_id")
                        pyr_l[i].target = id

                syn_spec_ff_pi = syn_laminar_pyr_intn
                syn_spec_ff_pi['weight'] = (
                    syn_spec_ff_pp['weight'] if self_predicting else self.gen_weights(self.dims[lr-1], self.dims[lr]))
                print(syn_spec_ff_pi['weight'])
                nest.Connect(pyr_prev, int_l, syn_spec=syn_spec_ff_pi)

                syn_spec_fb_pi = syn_laminar_intn_pyr
                syn_spec_fb_pi['weight'] = (-1 * syn_spec_fb_pp['weight']
                                            if self_predicting else self.gen_weights(self.dims[lr], self.dims[lr-1]))
                nest.Connect(int_l, pyr_prev, syn_spec=syn_spec_fb_pi)

                self.intn_pops.append(int_l)

        self.pyr_pops[-1].set({"apical_lat": {"g": 0}})

        self.nudge = nest.Create("dc_generator", self.dims[-1], {'amplitude': 0})
        nest.Connect(self.nudge, self.pyr_pops[-1], "one_to_one", syn_spec={'receptor_type': pyr_comps['soma_curr']})

        self.mm_pyr_0 = nest.Create('multimeter', 1, {'record_from': ["V_m.a_lat", "V_m.a_td", "V_m.b", "V_m.s"]})
        self.mm_intn = nest.Create('multimeter', 1, {'record_from': ["V_m.s"]})
        self.mm_pyr_1 = nest.Create('multimeter', 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_pyr_0, self.pyr_pops[1])
        nest.Connect(self.mm_pyr_1, self.pyr_pops[2])
        nest.Connect(self.mm_intn, self.intn_pops[0])

        self.sr_intn = nest.Create("spike_recorder", 1)
        self.sr_pyr = nest.Create("spike_recorder", 1)
        self.sr_out = nest.Create("spike_recorder", 1)
        nest.Connect(self.intn_pops[0], self.sr_intn)
        nest.Connect(self.pyr_pops[1], self.sr_pyr)
        nest.Connect(self.pyr_pops[-1], self.sr_out)

        self.stim = nest.Create("dc_generator", self.dims[0])

        nest.Connect(self.stim, self.pyr_pops[0], conn_spec="one_to_one",
                     syn_spec={"receptor_type": pyr_comps['soma_curr']})

        self.record_neuron = self.pyr_pops[1][0]
        self.record_id = self.record_neuron.get("global_id")

    def set_input(self, indices):
        for i in range(self.dims[0]):
            if i in indices:
                self.stim[i].amplitude = self.stim_amp
                # nudge[i].amplitude = target_amp
            else:
                self.stim[i].amplitude = 0
                # nudge[i].amplitude = -target_amp
