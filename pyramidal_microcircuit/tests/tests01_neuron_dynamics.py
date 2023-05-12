from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.test_utils import *

import nest


class FilteredInputCurrent(TestClass):
    # this test shows that a neuron with attenuated leakage conductance and injected
    # current behaves like a low pass filter on injected current. It also shows that setting
    # currents through a step generator leads to the same result, thus enabling batch learning
    # for slightly better performance.
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)

        self.neuron_01 = nest.Create(self.p.neuron_model, 1, params.input_params)
        self.mm_01 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
        nest.Connect(self.mm_01, self.neuron_01)

        self.neuron_02 = nest.Create(self.p.neuron_model, 1, params.input_params)
        self.mm_02 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
        nest.Connect(self.mm_02, self.neuron_02)
        compartments = nest.GetDefaults(self.p.neuron_model)["receptor_types"]
        self.step_generator = nest.Create("step_current_generator")
        nest.Connect(self.step_generator, self.neuron_02, syn_spec={"receptor_type": compartments["soma_curr"]})

        self.y = []

        self.sim_times = [50 for i in range(3)]
        self.stim_amps = [2, -2, 0]

    def run(self):
        delta_u = 0
        ux = 0
        self.step_generator.set(amplitude_values=np.array(self.stim_amps)/self.tau_x,
                                amplitude_times=np.cumsum(self.sim_times).astype(float) - 50 + self.delta_t)
        for T, amp in zip(self.sim_times, self.stim_amps):
            for i in range(int(T/self.delta_t)):
                delta_u = -ux + amp
                ux = ux + (self.delta_t/self.tau_x) * delta_u
                self.y.append(ux)

            self.neuron_01.set({"soma": {"I_e": amp/self.tau_x}})
            nest.Simulate(T)

    def evaluate(self) -> bool:
        return records_match(self.y, self.mm_01.get("events")["V_m.s"]) and records_match(self.y, self.mm_01.get("events")["V_m.s"])

    def plot_results(self):
        plt.plot(self.y, label="exact low pass filtering")
        plt.plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="pyramidal neuron with injected current")
        plt.plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="pyramidal neuron with dc generator")
        plt.title("Somatic voltage")
        plt.legend()


class TargetCurrent(FilteredInputCurrent):

    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)

        self.neuron_01.set(params.pyr_params)
        self.neuron_02.set(params.pyr_params)

    def run(self):
        delta_u = 0
        uy = 0
        self.step_generator.set(amplitude_values=np.array(self.stim_amps)*self.g_som,
                                amplitude_times=np.cumsum(self.sim_times).astype(float) - 50 + self.delta_t)
        for T, amp in zip(self.sim_times, self.stim_amps):
            for i in range(int(T/self.delta_t)):
                delta_u = -(self.g_l + self.g_d + self.g_a) * uy + amp * self.g_som
                uy = uy + (self.delta_t) * delta_u
                self.y.append(uy)

            self.neuron_01.set({"soma": {"I_e": amp*self.g_som}})
            nest.Simulate(T)

    def evaluate(self) -> bool:
        return records_match(self.mm_01.get("events")["V_m.s"], self.y) and records_match(self.mm_01.get("events")["V_m.s"], self.y)

    def plot_results(self):
        plt.plot(self.y, label="exact low pass filtering")
        plt.plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="pyramidal neuron with injected current")
        plt.plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="pyramidal neuron with dc generator")
        plt.title("Somatic voltage")
        plt.legend()


class CurrentConnection(TestClass):
    # This test shows that the current connection which transmits somatic voltage to a single target neuron
    # neuron model behaves as intended and causes appropriate changes in the somatic voltage.

    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)

        self.pyr_y = nest.Create(params.neuron_model, 1, params.input_params)
        self.mm_y = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_y, self.pyr_y)

        self.intn = nest.Create(params.neuron_model, 1, params.pyr_params)
        self.mm_i = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_i, self.intn)

        pyr_id = self.intn.get("global_id")
        self.pyr_y.target = pyr_id

    def run(self):
        U_i = 0
        U_y = 0

        sim_times = [50 for i in range(3)]
        stim_amps = [2, -2, 0]

        self.UI = []
        self.UY = []

        for T, amp in zip(sim_times, stim_amps):
            self.pyr_y.set({"soma": {"I_e": amp/self.tau_x}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):
                delta_u_y = -U_y + amp
                U_y = U_y + (self.delta_t/self.tau_x) * delta_u_y

                delta_u_i = -self.g_l_eff * U_i + self.g_som * U_y
                U_i = U_i + self.delta_t * delta_u_i
                self.UI.append(U_i)
                self.UY.append(U_y)

    def evaluate(self) -> bool:
        return records_match(self.UY, self.mm_y.events["V_m.s"], 0.05) and records_match(self.UI, self.mm_i.events["V_m.s"], 0.05)

    def plot_results(self):
        fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, constrained_layout=True)
        ax0.plot(*zip(*read_multimeter(self.mm_y, "V_m.s")), label="NEST")
        ax0.plot(self.UY, label="analytical")

        ax1.plot(*zip(*read_multimeter(self.mm_i, "V_m.s")), label="NEST")
        ax1.plot(self.UI, label="analytical")

        ax0.set_title("input neuron")
        ax1.set_title("output neuron")

        ax0.legend()
        ax1.legend()


class DynamicsHX(TestClass):
    """
    This test shows that the neuron model handles a single dendritic input exactly like the analytical
    solution if parameters are set correctly.
    """

    def __init__(self, params, **kwargs) -> None:

        super().__init__(params, **kwargs)
        params.setup_nest_configs()
        self.weight = 1

        conn_hx = self.p.syn_plastic
        conn_hx.update({"weight": self.weight / self.psi, "eta": 0,
                       "receptor_type": self.p.compartments["basal"]})

        self.neuron_01 = nest.Create(params.neuron_model, 1, params.input_params)
        self.mm_01 = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_01, self.neuron_01)

        self.neuron_02 = nest.Create(params.neuron_model, 1, params.pyr_params)
        self.mm_02 = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat", "V_m.a_td"]})
        nest.Connect(self.mm_02, self.neuron_02)
        nest.Connect(self.neuron_01, self.neuron_02, syn_spec=conn_hx)

        self.n_runs = 3
        self.sim_times = [100 for i in range(self.n_runs)]
        self.stim_amps = np.random.random(self.n_runs)
        self.SIM_TIME = sum(self.sim_times)

    def run(self):
        U_x = 0
        U_h = 0
        V_bh = 0
        self.UX = []
        self.UH = []
        self.VBH = []
        for T, amp in zip(self.sim_times, self.stim_amps):
            self.neuron_01.set({"soma": {"I_e": amp/self.tau_x}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):

                delta_u_x = -U_x + amp
                delta_u_h = -self.g_l_eff * U_h + V_bh * self.g_d

                U_x = U_x + (self.delta_t/self.tau_x) * delta_u_x
                r_x = U_x

                V_bh = r_x * self.weight
                U_h = U_h + self.delta_t * delta_u_h

                self.UX.append(U_x)
                self.UH.append(U_h)
                self.VBH.append(V_bh)

    def evaluate(self) -> bool:
        return records_match(self.VBH, self.mm_02.events["V_m.b"], 0.01) and records_match(self.UH, self.mm_02.events["V_m.s"], 0.01)

    def plot_results(self):

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, constrained_layout=True)

        ax0.plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST computed")
        ax0.plot(self.UX, label="analytical")

        ax1.plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST computed")
        ax1.plot(self.UH, label="analytical")

        ax2.plot(*zip(*read_multimeter(self.mm_02, "V_m.b")), label="NEST computed")
        ax2.plot(self.VBH, label="analytical")

        ax0.set_title("UX")
        ax1.set_title("UH")
        ax2.set_title("VBH")

        ax0.legend()
        ax1.legend()
        ax2.legend()


class DynamicsHXMulti(DynamicsHX):
    """
    This test shows that the neuron model handles a single dendritic input exactly like the analytical
    solution if parameters are set correctly.
    """

    def __init__(self, params, **kwargs) -> None:

        super().__init__(params, **kwargs)

        self.weight_2 = -0.5
        conn_hx = self.p.syn_plastic
        conn_hx["weight"] = self.weight_2 / self.psi

        self.neuron_03 = nest.Create(self.p.neuron_model, 1, self.p.input_params)
        self.mm_03 = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_03, self.neuron_03)

        nest.Connect(self.neuron_03, self.neuron_02, syn_spec=conn_hx)

    def run(self):
        U_x = 0
        U_x2 = 0
        U_h = 0
        V_bh = 0
        self.UX = []
        self.UX2 = []
        self.UH = []
        self.VBH = []
        self.stim_amps_2 = np.random.random(self.n_runs)
        for T, amp, amp2 in zip(self.sim_times, self.stim_amps, self.stim_amps_2):
            self.neuron_01.set({"soma": {"I_e": amp/self.tau_x}})
            self.neuron_03.set({"soma": {"I_e": amp2/self.tau_x}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):

                delta_u_x = -U_x + amp
                delta_u_x2 = -U_x2 + amp2

                delta_u_h = -self.g_l_eff * U_h + V_bh * self.g_d

                U_x = U_x + (self.delta_t/self.tau_x) * delta_u_x
                U_x2 = U_x2 + (self.delta_t/self.tau_x) * delta_u_x2
                r_x = U_x
                r_x2 = U_x2

                V_bh = r_x * self.weight + r_x2 * self.weight_2
                U_h = U_h + self.delta_t * delta_u_h

                self.UX.append(U_x)
                self.UX2.append(U_x2)
                self.UH.append(U_h)
                self.VBH.append(V_bh)

    def plot_results(self):

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharex=True, constrained_layout=True)

        ax0.plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST")
        ax0.plot(self.UX, label="target")

        ax1.plot(*zip(*read_multimeter(self.mm_03, "V_m.s")), label="NEST")
        ax1.plot(self.UX2, label="target")

        ax2.plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST")
        ax2.plot(self.UH, label="target")

        ax3.plot(*zip(*read_multimeter(self.mm_02, "V_m.b")), label="NEST")
        ax3.plot(self.VBH, label="target")

        ax0.set_title("UX1")
        ax1.set_title("UX2")
        ax2.set_title("UH")
        ax3.set_title("VBH")

        ax0.legend()
        ax1.legend()
        ax2.legend()


class DynamicsHY(DynamicsHX):
    """
    This test shows that the neuron model handles a single dendritic input exactly like the analytical
    solution if parameters are set correctly.
    """

    def __init__(self, params, **kwargs) -> None:

        super().__init__(params, **kwargs)

        synapse = self.p.syn_plastic
        synapse.update({"weight": self.weight, "eta": 0, "receptor_type": self.p.compartments["apical_lat"]})

        self.neuron_01.set(params.intn_params)
        self.neuron_02.set(params.pyr_params)
        nest.Disconnect(self.neuron_01, self.neuron_02, conn_spec='all_to_all',
                        syn_spec={'synapse_model': self.p.syn_plastic["synapse_model"]})

        nest.Connect(self.neuron_01, self.neuron_02, syn_spec=synapse)

    def run(self):
        U_i = 0
        r_i = 0
        U_h = 0
        V_ah = 0
        self.UI = []
        self.UH = []
        self.VAH = []

        for T, amp in zip(self.sim_times, self.stim_amps):
            self.neuron_01.set({"soma": {"I_e": amp}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):

                delta_u_i = -self.g_l_eff * U_i + amp
                delta_u_h = -self.g_l_eff * U_h + V_ah * self.g_a

                U_i = U_i + self.delta_t * delta_u_i
                r_i = self.phi(U_i)
                V_ah = r_i * self.weight
                U_h = U_h + self.delta_t * delta_u_h

                self.UI.append(U_i)
                self.UH.append(U_h)
                self.VAH.append(V_ah)

    def evaluate(self) -> bool:
        VAH_nest = self.mm_02.events["V_m.a_lat"]
        UH_nest = self.mm_02.events["V_m.s"]
        if self.spiking:
            return records_match(self.VAH, VAH_nest, 0.01) and records_match(self.UH, UH_nest, 0.01)
        else:
            return records_match(self.VAH, VAH_nest) and records_match(self.UH, UH_nest)

    def plot_results(self):

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, constrained_layout=True)

        ax0.plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST computed")
        ax0.plot(self.UI, label="analytical")

        ax1.plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST computed")
        ax1.plot(self.UH, label="analytical")

        ax2.plot(*zip(*read_multimeter(self.mm_02, "V_m.a_lat")), label="NEST computed")
        ax2.plot(self.VAH, label="analytical")

        ax0.set_title("UI")
        ax1.set_title("UH")
        ax2.set_title("VAH")

        ax0.legend()
        ax1.legend()
        ax2.legend()


class DynamicsHY(DynamicsHX):
    """
    This test shows that the neuron model handles a single dendritic input exactly like the analytical
    solution if parameters are set correctly.
    """

    def __init__(self, params, **kwargs) -> None:

        super().__init__(params, **kwargs)

        synapse = self.p.syn_plastic
        synapse.update({"weight": self.weight / self.psi, "eta": 0,
                       "receptor_type": self.p.compartments["apical_td"]})

        self.neuron_01.set(params.intn_params)
        self.neuron_02.set(params.pyr_params)
        nest.Disconnect(self.neuron_01, self.neuron_02, conn_spec='all_to_all',
                        syn_spec={'synapse_model': self.p.syn_plastic["synapse_model"]})

        nest.Connect(self.neuron_01, self.neuron_02, syn_spec=synapse)

    def run(self):
        U_y = 0
        r_y = 0
        U_h = 0
        V_ah = 0
        self.UY = []
        self.UH = []
        self.VAH = []

        for T, amp in zip(self.sim_times, self.stim_amps):
            self.neuron_01.set({"soma": {"I_e": amp}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):

                delta_u_y = -self.g_l_eff * U_y + amp
                delta_u_h = -self.g_l_eff * U_h + V_ah * self.g_a

                U_y = U_y + self.delta_t * delta_u_y
                r_y = self.phi(U_y)
                V_ah = r_y * self.weight
                U_h = U_h + self.delta_t * delta_u_h

                self.UY.append(U_y)
                self.UH.append(U_h)
                self.VAH.append(V_ah)

    def evaluate(self) -> bool:
        VAH_nest = self.mm_02.events["V_m.a_lat"]
        UH_nest = self.mm_02.events["V_m.s"]
        if self.spiking:
            return records_match(self.VAH, VAH_nest, 0.01) and records_match(self.UH, UH_nest, 0.01)
        else:
            return records_match(self.VAH, VAH_nest) and records_match(self.UH, UH_nest)

    def plot_results(self):

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharex=True, constrained_layout=True)

        ax0.plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST computed")
        ax0.plot(self.UY, label="analytical")

        ax1.plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST computed")
        ax1.plot(self.UH, label="analytical")

        ax2.plot(*zip(*read_multimeter(self.mm_02, "V_m.a_lat")), label="NEST computed")
        ax2.plot(self.VAH, label="analytical")

        ax3.plot(*zip(*read_multimeter(self.mm_02, "V_m.a_td")), label="NEST computed")

        ax0.set_title("UY")
        ax1.set_title("UH")
        ax2.set_title("VAH")
        ax3.set_title("VAH distal")
        ax0.legend()
        ax1.legend()
        ax2.legend()


class DynamicsYH(DynamicsHX):
    """
    This test shows that the neuron model handles a single dendritic input exactly like the analytical
    solution if parameters are set correctly.
    """

    def __init__(self, params, **kwargs) -> None:

        super().__init__(params, **kwargs)

        synapse = self.p.syn_plastic
        synapse.update({"weight": self.weight, "eta": 0, "receptor_type": self.p.compartments["basal"]})

        self.neuron_01.set(params.pyr_params)
        self.neuron_02.set(params.pyr_params)
        nest.Disconnect(self.neuron_01, self.neuron_02, conn_spec='all_to_all',
                        syn_spec={'synapse_model': self.p.syn_plastic["synapse_model"]})

        nest.Connect(self.neuron_01, self.neuron_02, syn_spec=synapse)

    def run(self):
        U_h = 0
        U_y = 0
        V_bh = 0
        self.UH = []
        self.UY = []
        self.VBY = []

        for T, amp in zip(self.sim_times, self.stim_amps):
            self.neuron_01.set({"soma": {"I_e": amp}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):

                delta_u_i = -self.g_l_eff * U_h + amp
                delta_u_h = -self.g_l_eff * U_y + V_bh * self.g_d

                U_h = U_h + self.delta_t * delta_u_i
                V_bh = self.phi(U_h) * self.weight
                U_y = U_y + self.delta_t * delta_u_h

                self.UH.append(U_h)
                self.UY.append(U_y)
                self.VBY.append(V_bh)

    def evaluate(self) -> bool:
        VBH_nest = self.mm_02.events["V_m.b"]
        UH_nest = self.mm_02.events["V_m.s"]
        if self.spiking:
            return records_match(self.VBY, VBH_nest, 0.01) and records_match(self.UY, UH_nest, 0.01)
        else:
            return records_match(self.VBY, VBH_nest) and records_match(self.UY, UH_nest)

    def plot_results(self):

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, constrained_layout=True)

        ax0.plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST computed")
        ax0.plot(self.UH, label="analytical")

        ax1.plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST computed")
        ax1.plot(self.UY, label="analytical")

        ax2.plot(*zip(*read_multimeter(self.mm_02, "V_m.b")), label="NEST computed")
        ax2.plot(self.VBY, label="analytical")

        ax0.set_title("UH")
        ax1.set_title("UY")
        ax2.set_title("VBY")

        ax0.legend()
        ax1.legend()
        ax2.legend()


class NetworkDynamics(TestClass):

    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.disable_plasticity()
        self.numpy_net = NumpyNetwork(deepcopy(params))
        params.setup_nest_configs()
        self.nest_net = NestNetwork(deepcopy(params))
        self.numpy_net.set_all_weights(self.nest_net.get_weight_dict())

    def run(self):
        self.t_pres = 10
        for i in range(5):
            input_currents = np.random.random(self.dims[0])
            target_currents = np.random.random(self.dims[-1])

            self.nest_net.set_input(input_currents)
            self.nest_net.set_target(target_currents)
            self.nest_net.simulate(self.t_pres, enable_recording=True, with_delay=False)

            self.numpy_net.set_input(input_currents)
            self.numpy_net.set_target(target_currents)
            self.numpy_net.simulate(self.t_pres, enable_recording=True, plasticity=False)

    def evaluate(self) -> bool:
        records = pd.DataFrame.from_dict(self.nest_net.mm.events)
        self.nest_UH = records[records["senders"].isin(self.nest_net.layers[-2].pyr.global_id)].sort_values(
            ["senders", "times"])["V_m.s"].values.reshape((self.dims[-2], -1)).swapaxes(0, 1)
        self.nest_VAH = records[records["senders"].isin(self.nest_net.layers[-2].pyr.global_id)].sort_values(
            ["senders", "times"])["V_m.a_lat"].values.reshape((self.dims[-2], -1)).swapaxes(0, 1)
        self.nest_VBH = records[records["senders"].isin(self.nest_net.layers[-2].pyr.global_id)].sort_values(
            ["senders", "times"])["V_m.b"].values.reshape((self.dims[-2], -1)).swapaxes(0, 1)
        self.nest_UI = records[records["senders"].isin(self.nest_net.layers[-2].intn.global_id)].sort_values(
            ["senders", "times"])["V_m.s"].values.reshape((self.dims[-1], -1)).swapaxes(0, 1)
        self.nest_UY = records[records["senders"].isin(self.nest_net.layers[-1].pyr.global_id)].sort_values(
            ["senders", "times"])["V_m.s"].values.reshape((self.dims[-1], -1)).swapaxes(0, 1)
        return records_match(self.nest_UH, np.asarray(self.numpy_net.U_h_record)) and \
            records_match(self.nest_VAH, np.asarray(self.numpy_net.V_ah_record)) and \
            records_match(self.nest_UI, np.asarray(self.numpy_net.U_i_record)) and \
            records_match(self.nest_UY, np.asarray(self.numpy_net.U_y_record))

    def plot_results(self):
        fig, axes = plt.subplots(2, 3, sharex=True, constrained_layout=True)
        cmap = plt.cm.get_cmap('hsv', max(self.dims)+1)

        for i in range(self.dims[0]):
            axes[0][0].plot(self.numpy_net.U_x_record[:, i], color=cmap(i))
            # axes[0][0].plot(self.nest_UX[:, i], color=cmap(i), linestyle="dashed", alpha=0.7)

        for i in range(self.dims[-2]):
            axes[0][1].plot(self.numpy_net.V_bh_record[:, i], color=cmap(i))
            axes[0][1].plot(self.nest_VBH[:, i], color=cmap(i), linestyle="dashed", alpha=0.7)

            axes[0][2].plot(self.numpy_net.U_h_record[:, i], color=cmap(i))
            axes[0][2].plot(self.nest_UH[:, i], color=cmap(i), linestyle="dashed", alpha=0.7)

            axes[1][0].plot(self.numpy_net.V_ah_record[:, i], color=cmap(i))
            axes[1][0].plot(self.nest_VAH[:, i], color=cmap(i), linestyle="dashed", alpha=0.7)

        for i in range(self.dims[-1]):

            axes[1][1].plot(self.numpy_net.U_i_record[:, i], color=cmap(i))
            axes[1][1].plot(self.nest_UI[:, i], color=cmap(i), linestyle="dashed", alpha=0.7)

            axes[1][2].plot(self.numpy_net.U_y_record[:, i], color=cmap(i))
            axes[1][2].plot(self.nest_UY[:, i], color=cmap(i), linestyle="dashed", alpha=0.7)

        axes[0][0].set_title("UX")
        axes[0][1].set_title("VBH")
        axes[0][2].set_title("UH")
        axes[1][0].set_title("VAH")
        axes[1][1].set_title("UI")
        axes[1][2].set_title("UY")

        fig.suptitle("dashed: NEST computed")

        # if not self.spiking:
        #     plt.show()
        # axes[0][0].set_ylabel("NEST computed")
        # axes[1][0].set_ylabel("Target activation")


class DeepNetworkDynamics(NetworkDynamics):

    def __init__(self, params, **kwargs) -> None:
        params.dims = [4, 3, 2, 2]
        params.eta = {
            "ip": [
                0.004,
                0.004,
                0.0
            ],
            "pi": [
                0.01,
                0.01,
                0.0
            ],
            "up": [
                0.01,
                0.01,
                0.003
            ],
            "down": [
                0,
                0,
                0
            ]
        }
        super().__init__(params, **kwargs)
