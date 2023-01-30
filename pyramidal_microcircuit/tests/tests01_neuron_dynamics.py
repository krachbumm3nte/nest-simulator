import matplotlib.pyplot as plt
import nest
import numpy as np
from test_utils import *


class FilteredInputCurrent(TestClass):
    # this test shows that a neuron with attenuated leakage conductance and injected
    # current behaves like a low pass filter on injected current. It also shows that setting
    # currents through a step generator leads to the same result, thus enabling batch learning
    # for slightly better performance.
    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)

        self.neuron_01 = nest.Create(self.neuron_model, 1, nrn["input"])
        self.mm_01 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
        nest.Connect(self.mm_01, self.neuron_01)

        self.neuron_02 = nest.Create(self.neuron_model, 1, nrn["input"])
        self.mm_02 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
        nest.Connect(self.mm_02, self.neuron_02)
        compartments = nest.GetDefaults(self.neuron_model)["receptor_types"]
        step_generator = nest.Create("step_current_generator")
        nest.Connect(step_generator, self.neuron_02, syn_spec={"receptor_type": compartments["soma_curr"]})

        delta_u = 0
        ux = 0
        self.y = []

        sim_times = [50 for i in range(3)]
        stim_amps = [2, -2, 0]

        step_generator.set(amplitude_values=np.array(stim_amps)/self.tau_x,
                           amplitude_times=np.cumsum(sim_times).astype(float) - 50 + self.delta_t)
        for T, amp in zip(sim_times, stim_amps):
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


class TargetCurrent(TestClass):

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)

        self.neuron_01 = nest.Create(self.neuron_model, 1, nrn["pyr"])
        self.mm_01 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
        nest.Connect(self.mm_01, self.neuron_01)

        self.neuron_02 = nest.Create(self.neuron_model, 1, nrn["pyr"])
        self.mm_02 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
        nest.Connect(self.mm_02, self.neuron_02)
        compartments = nest.GetDefaults(self.neuron_model)["receptor_types"]
        step_generator = nest.Create("step_current_generator")
        nest.Connect(step_generator, self.neuron_02, syn_spec={"receptor_type": compartments["soma_curr"]})

        delta_u = 0
        uy = 0
        self.y = []

        sim_times = [50 for i in range(3)]
        stim_amps = [2, -2, 0]

        step_generator.set(amplitude_values=np.array(stim_amps)*self.g_si,
                           amplitude_times=np.cumsum(sim_times).astype(float) - 50 + self.delta_t)
        for T, amp in zip(sim_times, stim_amps):
            for i in range(int(T/self.delta_t)):
                delta_u = -(self.g_l + self.g_d + self.g_a) * uy + amp * self.g_si
                uy = uy + (self.delta_t) * delta_u
                self.y.append(uy)

            self.neuron_01.set({"soma": {"I_e": amp*self.g_si}})
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

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)

        pyr_y = nest.Create(nrn["model"], 1, nrn["input"])
        self.mm_y = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_y, pyr_y)

        intn = nest.Create(nrn["model"], 1, nrn["pyr"])
        self.mm_i = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_i, intn)

        pyr_id = intn.get("global_id")
        pyr_y.target = pyr_id

        U_i = 0
        U_y = 0

        sim_times = [50 for i in range(3)]
        stim_amps = [2, -2, 0]
        SIM_TIME = sum(sim_times)

        self.UI = []
        self.UY = []

        for T, amp in zip(sim_times, stim_amps):
            pyr_y.set({"soma": {"I_e": amp/nrn["tau_x"]}})
            nest.Simulate(T)
            for i in range(int(T/sim["delta_t"])):
                delta_u_y = -U_y + amp
                U_y = U_y + (sim["delta_t"]/nrn["tau_x"]) * delta_u_y

                delta_u_i = -self.g_l_eff * U_i + nrn["g_si"] * U_y
                U_i = U_i + sim["delta_t"] * delta_u_i
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


class SingleCompartmentDynamics(TestClass):
    """
    This test shows that the neuron model handles a single dendritic input exactly like the analytical
    solution if parameters are set correctly.
    """

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)

        weight = 1
        syn["hx"].update({"weight": weight, "eta": 0})

        if spiking_neurons:
            weight_scale = 15000
            nrn["input"]["gamma"] = self.gamma * weight_scale/self.gamma
            syn["hx"].update({"weight": weight/weight_scale})
        

        print(self.gamma, nrn["input"]["gamma"], nrn["gamma"])
        pyr_x = nest.Create(self.neuron_model, 1, nrn["input"])
        self.mm_x = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_x, pyr_x)

        pyr_h = nest.Create(self.neuron_model, 1, nrn["pyr"])
        self.mm_h = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
        nest.Connect(self.mm_h, pyr_h)

        nest.Connect(pyr_x, pyr_h, syn_spec=syn["hx"])

        n_runs = 10
        sim_times = [150 for i in range(n_runs)]
        stim_amps = np.random.random(n_runs)
        SIM_TIME = sum(sim_times)

        U_x = 0
        U_h = 0
        V_bh = 0
        self.UX = []
        self.UH = []
        self.VBH = []
        for T, amp in zip(sim_times, stim_amps):
            pyr_x.set({"soma": {"I_e": amp/self.tau_x}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):

                delta_u_x = -U_x + amp
                delta_u_h = -self.g_l_eff * U_h + V_bh * self.g_d
                
                U_x = U_x + (self.delta_t/self.tau_x) * delta_u_x
                r_x = U_x

                V_bh = r_x * weight
                U_h = U_h + self.delta_t * delta_u_h



                self.UX.append(U_x)
                self.UH.append(U_h)
                self.VBH.append(V_bh)

        print(len(self.wr.events["times"]))

    def evaluate(self) -> bool:
        return records_match(self.VBH, self.mm_h.events["V_m.b"]) and records_match(self.UH, self.mm_h.events["V_m.s"])

    def plot_results(self):

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, constrained_layout=True)

        ax0.plot(*zip(*read_multimeter(self.mm_x, "V_m.s")), label="NEST computed")
        ax0.plot(self.UX, label="analytical")

        ax1.plot(*zip(*read_multimeter(self.mm_h, "V_m.s")), label="NEST computed")
        ax1.plot(self.UH, label="analytical")

        ax2.plot(*zip(*read_multimeter(self.mm_h, "V_m.b")), label="NEST computed")
        ax2.plot(self.VBH, label="analytical")

        ax0.set_title("input neuron voltage")
        ax1.set_title("output neuron somatic voltage")
        ax2.set_title("output neuron basal voltage")

        ax0.legend()
        ax1.legend()
        ax2.legend()



class SingleCompartmentDynamics2(TestClass):
    """
    This test shows that the neuron model handles a single dendritic input exactly like the analytical
    solution if parameters are set correctly.
    """

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:

        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)

        weight = 1
        syn["hi"].update({"weight": weight, "eta": 0})

        if spiking_neurons:
            weight_scale = 15000
            nrn["intn"]["gamma"] = self.gamma * weight_scale
            syn["hi"].update({"weight": weight/weight_scale})
        
        intn = nest.Create(self.neuron_model, 1, nrn["intn"])
        self.mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_in, intn)

        pyr_h = nest.Create(self.neuron_model, 1, nrn["pyr"])
        self.mm_h = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
        nest.Connect(self.mm_h, pyr_h)

        nest.Connect(intn, pyr_h, syn_spec=syn["hi"])

        n_runs = 5
        sim_times = [50 for i in range(n_runs)]
        stim_amps = np.random.random(n_runs)
        SIM_TIME = sum(sim_times)

        U_i = 0
        U_h = 0
        V_ah = 0
        self.UI = []
        self.UH = []
        self.VAH = []

        for T, amp in zip(sim_times, stim_amps):
            intn.set({"soma": {"I_e": amp}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):

                delta_u_i = -self.g_l_eff * U_i + amp
                delta_u_h = -self.g_l_eff * U_h + V_ah * self.g_a
                
                U_i = U_i + self.delta_t * delta_u_i
                V_ah = self.phi(U_i) * weight
                U_h = U_h + self.delta_t * delta_u_h



                self.UI.append(U_i)
                self.UH.append(U_h)
                self.VAH.append(V_ah)

        print(len(self.wr.events["times"]))

    def evaluate(self) -> bool:
        return records_match(self.VAH, self.mm_h.events["V_m.a_lat"]) and records_match(self.UH, self.mm_h.events["V_m.s"])

    def plot_results(self):

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, constrained_layout=True)

        ax0.plot(*zip(*read_multimeter(self.mm_in, "V_m.s")), label="NEST computed")
        ax0.plot(self.UI, label="analytical")

        ax1.plot(*zip(*read_multimeter(self.mm_h, "V_m.s")), label="NEST computed")
        ax1.plot(self.UH, label="analytical")

        ax2.plot(*zip(*read_multimeter(self.mm_h, "V_m.a_lat")), label="NEST computed")
        ax2.plot(self.VAH, label="analytical")

        ax0.set_title("UI")
        ax1.set_title("UH")
        ax2.set_title("VAH")

        ax0.legend()
        ax1.legend()
        ax2.legend()





class NetworkDynamics(TestClass):

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        for syn_name in ["hx", "yh", "hy", "hi", "ih"]:
            if "eta" in syn[syn_name]:
                syn[syn_name]["eta"] = 0
        sim["teacher"] = False
        sim["noise"] = False
        sim["dims"] = [4, 3, 2]

        if spiking_neurons:
            weight_scale = nrn["weight_scale"]
            nrn["input"]["gamma"] = weight_scale
            nrn["pyr"]["gamma"] = weight_scale * nrn["pyr"]["gamma"]
            nrn["intn"]["gamma"] = weight_scale * nrn["intn"]["gamma"]
            syn["wmin_init"] = -1/weight_scale
            syn["wmax_init"] = 1/weight_scale
        else:
            weight_scale = 1
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)

        self.nest_net = NestNetwork(sim, nrn, syn)
        self.numpy_net = NumpyNetwork(sim, nrn, syn)

        input_currents = np.random.random(self.dims[0])
        target_currents = np.random.random(self.dims[2])

        self.nest_net.set_input(input_currents)
        self.nest_net.set_target(target_currents)

        self.numpy_net.set_input(input_currents)
        self.numpy_net.set_target(target_currents)

        weights = self.nest_net.get_weight_dict()

        for conn in ["hi", "ih", "hx", "hy", "yh"]:
            self.numpy_net.conns[conn]["w"] = weights[conn] * weight_scale

        sim_time = 100
        self.nest_net.simulate(sim_time)
        for i in range(int(sim_time/self.delta_t)):
            self.numpy_net.simulate(self.numpy_net.train_static)

    def evaluate(self) -> bool:
        records = pd.DataFrame.from_dict(self.nest_net.mm.events)

        self.nest_UH = records[records["senders"].isin(self.nest_net.pyr_pops[1].global_id)].sort_values(
            ["senders", "times"])["V_m.s"].values.reshape((self.dims[1], -1)).swapaxes(0, 1)
        self.nest_VAH = records[records["senders"].isin(self.nest_net.pyr_pops[1].global_id)].sort_values(
            ["senders", "times"])["V_m.a_lat"].values.reshape((self.dims[1], -1)).swapaxes(0, 1)
        self.nest_UI = records[records["senders"].isin(self.nest_net.intn_pops[0].global_id)].sort_values(
            ["senders", "times"])["V_m.s"].values.reshape((self.dims[-1], -1)).swapaxes(0, 1)
        self.nest_UY = records[records["senders"].isin(self.nest_net.pyr_pops[-1].global_id)].sort_values(
            ["senders", "times"])["V_m.s"].values.reshape((self.dims[-1], -1)).swapaxes(0, 1)
        self.nest_UX = records[records["senders"].isin(self.nest_net.pyr_pops[0].global_id)].sort_values(
            ["senders", "times"])["V_m.s"].values.reshape((self.dims[0], -1)).swapaxes(0, 1)

        return records_match(self.nest_UH, np.asarray(self.numpy_net.U_h_record)) and \
            records_match(self.nest_VAH, np.asarray(self.numpy_net.V_ah_record)) and \
            records_match(self.nest_UI, np.asarray(self.numpy_net.U_i_record)) and \
            records_match(self.nest_UY, np.asarray(self.numpy_net.U_y_record)) and \
            records_match(self.nest_UX, np.asarray(self.numpy_net.U_x_record))

    def plot_results(self):
        fig, axes = plt.subplots(2, 5, sharex=True, sharey="col", constrained_layout=True)
        cmap = plt.cm.get_cmap('hsv', max(self.dims)+1)

        for i in range(self.dims[0]):
            axes[0][0].plot(self.nest_UX[:, i], color=cmap(i))
            axes[1][0].plot(self.numpy_net.U_x_record[:, i], color=cmap(i))

        for i in range(self.dims[1]):
            axes[0][1].plot(self.nest_UH[:, i], color=cmap(i))
            axes[1][1].plot(self.numpy_net.U_h_record[:, i], color=cmap(i))

            axes[0][2].plot(self.nest_VAH[:, i], color=cmap(i))
            axes[1][2].plot(self.numpy_net.V_ah_record[:, i], color=cmap(i))

        for i in range(self.dims[2]):
            axes[0][3].plot(self.nest_UY[:, i], color=cmap(i))
            axes[1][3].plot(self.numpy_net.U_y_record[:, i], color=cmap(i))

            axes[0][4].plot(self.nest_UI[:, i], color=cmap(i))
            axes[1][4].plot(self.numpy_net.U_i_record[:, i], color=cmap(i))

        axes[0][0].set_title("UX")
        axes[0][1].set_title("UH")
        axes[0][2].set_title("VAH")
        axes[0][3].set_title("UY")
        axes[0][4].set_title("UI")
        axes[0][0].set_ylabel("NEST computed")
        axes[1][0].set_ylabel("Target activation")


class Dummy(TestClass):

    def __init__(self, nrn, sim, syn, **kwargs) -> None:
        super().__init__(nrn, sim, syn, **kwargs)

    def evaluate(self) -> bool:
        pass

    def plot_results(self):
        pass
