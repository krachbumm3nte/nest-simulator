from test_utils import *
from tests01_neuron_dynamics import *
from pprint import pprint


class PlasticityYH(DynamicsYH):

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, record_weights=True, **kwargs)
        self.eta = 0.04

        conn = nest.GetConnections(self.neuron_01, self.neuron_02)
        if spiking_neurons:
            conn.eta = self.eta/(self.weight_scale**2 * self.syn["tau_Delta"])
            conn.weight = self.weight / self.weight_scale
        else:
            conn.eta = self.eta

    def run(self):
        U_h = 0
        r_h = 0
        V_by = 0
        U_y = 0
        r_y = 0
        delta_tilde_w = 0
        tilde_w = 0
        self.UH = []
        self.UY = []
        self.VBY = []
        self.weights_numpy = []
        
        self.conn = nest.GetConnections(self.neuron_01, self.neuron_02)
        self.weights_nest = [self.conn.weight]

        for i, (T, amp) in enumerate(zip(self.sim_times, self.stim_amps)):

            self.neuron_01.set({"soma": {"I_e": amp}})
            
            for i in range(int(T/self.delta_t)):
                nest.Simulate(self.delta_t)
                self.weights_nest.append(self.conn.weight)

                delta_u_x = -self.g_l_eff * U_h + amp
                delta_u_h = -self.g_l_eff * U_y + V_by * self.g_d

                delta_tilde_w = -tilde_w + (r_y - self.phi(V_by * self.lambda_out)) * r_h

                U_h = U_h + self.delta_t * delta_u_x
                r_h = self.phi(U_h)

                V_by = r_h * self.weight
                U_y = U_y + self.delta_t * delta_u_h
                r_y = self.phi(U_y)

                tilde_w = tilde_w + (self.delta_t/self.tau_delta) * delta_tilde_w
                self.weight = np.clip(self.weight + self.eta * self.delta_t * tilde_w, -4, 4)

                self.UH.append(U_h)
                self.UY.append(U_y)
                self.VBY.append(V_by)
                self.weights_numpy.append(self.weight)


    def evaluate(self) -> bool:
        if self.spiking_neurons:
            self.weights_nest = [val*self.weight_scale for val in self.weights_nest]

        return records_match(self.weights_nest, self.weights_numpy, 0.1)

    def plot_results(self):
        fig, axes = plt.subplots(1, 4, sharex=True, constrained_layout=True)

        axes[0].plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST")
        axes[0].plot(self.UH, label="target")
        axes[0].set_title("UH")

        axes[1].plot(*zip(*read_multimeter(self.mm_02, "V_m.b")), label="NEST")
        axes[1].plot(self.VBY, label="target")
        axes[1].set_title("VBY")

        axes[2].plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST")
        axes[2].plot(self.UY, label="target")
        axes[2].set_title("UY")

        axes[3].plot(self.weights_nest, label="NEST")
        axes[3].plot(self.weights_numpy, label="target")
        axes[3].legend()
        axes[3].set_title("synaptic weight")


class PlasticityHX(DynamicsHX):

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, record_weights=True, **kwargs)
        self.eta = 0.04

        self.conn = nest.GetConnections(self.neuron_01, self.neuron_02)
        if spiking_neurons:
            self.conn.eta = self.eta/(self.weight_scale**2 * self.syn["tau_Delta"])
            self.conn.weight = self.weight / self.weight_scale
        else:
            self.conn.eta = self.eta

    def run(self):
        U_x = 0
        r_x = 0
        V_bh = 0
        U_h = 0
        r_h = 0
        delta_tilde_w = 0
        tilde_w = 0
        self.UX = []
        self.UH = []
        self.VBH = []
        self.weights_numpy = []
        self.weights_nest = []

        for i, (T, amp) in enumerate(zip(self.sim_times, self.stim_amps)):
            self.neuron_01.set({"soma": {"I_e": amp/self.tau_x}})

            for i in range(int(T/self.delta_t)):
                nest.Simulate(self.delta_t)
                self.weights_nest.append(self.conn.weight)

                delta_u_x = -U_x + amp
                delta_u_h = -self.g_l_eff * U_h + V_bh * self.g_d

                delta_tilde_w = -tilde_w + (r_h - self.phi(V_bh * self.lambda_bh)) * r_x

                U_x = U_x + (self.delta_t/self.tau_x) * delta_u_x
                r_x = U_x

                V_bh = r_x * self.weight
                U_h = U_h + self.delta_t * delta_u_h
                r_h = self.phi(U_h)

                tilde_w = tilde_w + (self.delta_t/self.tau_delta) * delta_tilde_w
                self.weight = self.weight + self.eta * self.delta_t * tilde_w
                self.weights_numpy.append(self.weight)

                self.UX.append(U_x)
                self.UH.append(U_h)
                self.VBH.append(V_bh)

    def evaluate(self) -> bool:
        if self.spiking_neurons:
            self.weights_nest = [val*self.weight_scale for val in self.weights_nest]
        return records_match(self.weights_nest, self.weights_numpy)

    def plot_results(self):
        fig, axes = plt.subplots(1, 4, sharex=True, constrained_layout=True)

        axes[0].plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST")
        axes[0].plot(self.UX, label="target")
        axes[0].set_title("UX")

        axes[1].plot(*zip(*read_multimeter(self.mm_02, "V_m.b")), label="NEST")
        axes[1].plot(self.VBH, label="target")
        axes[1].set_title("VBH")

        axes[2].plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST")
        axes[2].plot(self.UH, label="target")
        axes[2].set_title("UH")

        axes[3].plot(self.weights_nest, label="NEST")
        axes[3].plot(self.weights_numpy, label="target")
        axes[3].legend()
        axes[3].set_title("synaptic weight")


class PlasticityHXMulti(PlasticityHX):
    """
    This test shows that the neuron model handles a single dendritic input exactly like the analytical
    solution if parameters are set correctly.
    """

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:

        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)

        self.weight2 = -0.5
        if spiking_neurons:
            syn["conns"][0]["up"]["weight"] = self.weight2 / self.weight_scale
            syn["conns"][0]["up"]["eta"] = self.eta/(self.weight_scale**2 * syn["tau_Delta"])
        else:
            syn["conns"][0]["up"]["weight"] = self.weight2
            syn["conns"][0]["up"]["eta"] = self.eta

        self.neuron_03 = nest.Create(self.neuron_model, 1, nrn["input"])
        self.mm_03 = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
        nest.Connect(self.mm_03, self.neuron_03)

        nest.Connect(self.neuron_03, self.neuron_02, syn_spec=syn["conns"][0]["up"])

    def run(self):
        r_h = 0
        delta_tilde_w = 0
        delta_tilde_w2 = 0
        tilde_w = 0
        tilde_w2 = 0
        U_x = 0
        U_x2 = 0
        U_h = 0
        V_bh = 0
        r_x = 0
        r_x2 = 0
        self.UX = []
        self.UX2 = []
        self.UH = []
        self.VBH = []
        self.weights_numpy = []
        self.weights_numpy_2 = []
        self.conn2 = nest.GetConnections(self.neuron_03, self.neuron_02)
        self.weights_nest = []
        self.weights_nest_2 = [self.conn2.weight]

        self.stim_amps_2 = np.random.random(self.n_runs)
        for i, (T, amp, amp2) in enumerate(zip(self.sim_times, self.stim_amps, self.stim_amps_2)):
            self.neuron_01.set({"soma": {"I_e": amp/self.tau_x}})
            self.neuron_03.set({"soma": {"I_e": amp2/self.tau_x}})
            for i in range(int(T/self.delta_t)):
                nest.Simulate(self.delta_t)
                self.weights_nest.append(self.conn.weight)
                self.weights_nest_2.append(self.conn2.weight)

                delta_u_x = -U_x + amp
                delta_u_x2 = -U_x2 + amp2

                delta_u_h = -self.g_l_eff * U_h + V_bh * self.g_d

                delta_tilde_w = -tilde_w + (r_h - self.phi(V_bh * self.lambda_bh)) * r_x
                delta_tilde_w2 = -tilde_w2 + (r_h - self.phi(V_bh * self.lambda_bh)) * r_x2

                U_x = U_x + (self.delta_t/self.tau_x) * delta_u_x
                U_x2 = U_x2 + (self.delta_t/self.tau_x) * delta_u_x2
                r_x = U_x
                r_x2 = U_x2

                V_bh = r_x * self.weight + r_x2 * self.weight2
                U_h = U_h + self.delta_t * delta_u_h
                r_h = self.phi(U_h)

                tilde_w = tilde_w + (self.delta_t/self.tau_delta) * delta_tilde_w
                tilde_w2 = tilde_w2 + (self.delta_t/self.tau_delta) * delta_tilde_w2
                self.weight = self.weight + self.eta * self.delta_t * tilde_w
                self.weight2 = self.weight2 + self.eta * self.delta_t * tilde_w2
                self.weights_numpy.append(self.weight)
                self.weights_numpy_2.append(self.weight2)

                self.UX.append(U_x)
                self.UX2.append(U_x2)
                self.UH.append(U_h)
                self.VBH.append(V_bh)

    def evaluate(self) -> bool:
        if self.spiking_neurons:
            self.weights_nest = [val*self.weight_scale for val in self.weights_nest]
            self.nest_weights_2 = [val*self.weight_scale for val in self.weights_nest_2]

        return records_match(self.weights_nest, self.weights_numpy) and records_match(self.weights_nest_2, self.weights_numpy_2)

    def plot_results(self):

        fig, ([ax0, ax1, ax2], [ax3, ax4, ax5]) = plt.subplots(2, 3, sharex=True, constrained_layout=True)

        ax0.plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST")
        ax0.plot(self.UX, label="target")

        ax1.plot(*zip(*read_multimeter(self.mm_03, "V_m.s")), label="NEST")
        ax1.plot(self.UX2, label="target")

        ax2.plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST")
        ax2.plot(self.UH, label="target")

        ax3.plot(*zip(*read_multimeter(self.mm_02, "V_m.b")), label="NEST")
        ax3.plot(self.VBH, label="target")

        ax4.plot(self.weights_nest, label="NEST")
        ax4.plot(self.weights_numpy, label="target")

        ax5.plot(self.weights_nest_2, label="NEST")
        ax5.plot(self.weights_numpy_2, label="target")

        ax0.set_title("UX1")
        ax1.set_title("UX2")
        ax2.set_title("UH")
        ax3.set_title("VBH")
        ax4.set_title("weight 1")
        ax5.set_title("weight 2")

        ax0.legend()
        ax1.legend()
        ax2.legend()
        ax4.legend()
        ax5.legend()


class PlasticityIH(DynamicsHI):
    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, record_weights=True, **kwargs)
        self.eta = 0.04

        self.conn = nest.GetConnections(self.neuron_01, self.neuron_02)
        if spiking_neurons:
            self.conn.eta = self.eta/(self.weight_scale**2 * self.syn["tau_Delta"])
            self.conn.weight = self.weight / self.weight_scale
        else:
            self.conn.eta = self.eta

    def run(self):
        U_i = 0
        r_i = 0
        V_ah = 0
        U_h = 0
        delta_tilde_w = 0
        tilde_w = 0
        self.UX = []
        self.UH = []
        self.VBH = []
        self.weights_numpy = []

        self.weights_nest = []


        for i, (T, amp) in enumerate(zip(self.sim_times, self.stim_amps)):
            self.neuron_01.set({"soma": {"I_e": amp}})
            nest.SetKernelStatus({"data_prefix": f"it{str(i).zfill(8)}_"})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):
                nest.Simulate(self.delta_t)
                self.weights_nest.append(self.conn.weight)
                delta_tilde_w = -tilde_w - V_ah * r_i

                delta_u_i = -self.g_l_eff * U_i + amp
                delta_u_h = -self.g_l_eff * U_h + V_ah * self.g_a

                U_i = U_i + self.delta_t * delta_u_i
                r_i = self.phi(U_i)
                V_ah = r_i * self.weight
                U_h = U_h + self.delta_t * delta_u_h

                tilde_w = tilde_w + (self.delta_t/self.tau_delta) * delta_tilde_w
                self.weight = self.weight + self.eta * self.delta_t * tilde_w
                self.weights_numpy.append(self.weight)

                self.UX.append(U_i)
                self.UH.append(U_h)
                self.VBH.append(V_ah)

    def evaluate(self) -> bool:
        if self.spiking_neurons:
            self.weights_nest = [val*self.weight_scale for val in self.weights_nest]

        return records_match(self.weights_nest, self.weights_numpy)

    def plot_results(self):
        fig, axes = plt.subplots(1, 4, sharex=True, constrained_layout=True)

        axes[0].plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST")
        axes[0].plot(self.UX, label="target")
        axes[0].set_title("UI")

        axes[1].plot(*zip(*read_multimeter(self.mm_02, "V_m.a_lat")), label="NEST")
        axes[1].plot(self.VBH, label="target")
        axes[1].set_title("VAH")

        axes[2].plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST")
        axes[2].plot(self.UH, label="target")
        axes[2].set_title("UH")

        axes[3].plot(self.weights_nest, label="NEST")
        axes[3].plot(self.weights_numpy, label="target")
        axes[3].legend()
        axes[3].set_title("synaptic weight")


class NetworkPlasticity(TestClass):
    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        sim["teacher"] = False
        sim["noise"] = False
        sim["dims"] = [4, 3, 2]
        super().__init__(nrn, sim, syn, spiking_neurons, record_weights=True, **kwargs)

        self.numpy_net = NumpyNetwork(deepcopy(sim), deepcopy(nrn), deepcopy(syn))
        self.nest_net = NestNetwork(deepcopy(sim), deepcopy(nrn), deepcopy(syn), self.spiking_neurons)
        self.numpy_net.set_weights(self.nest_net.get_weight_dict())

    def run(self):
        

        self.sim_time = 100

        self.up_0 = []
        self.pi_0 = []
        self.ip_0 = []
        self.down_0 = []
        self.up_1 = []
        for i in range(15):
            input_currents = np.random.random(self.dims[0])
            target_currents = np.random.random(self.dims[-1])
            self.nest_net.set_input(input_currents)
            self.numpy_net.set_input(input_currents)
            self.nest_net.set_target(target_currents)
            for i in range(int(self.sim_time/self.delta_t)):
                self.numpy_net.simulate(lambda: target_currents)
                self.nest_net.simulate(self.delta_t)
                wgts = self.nest_net.get_weight_dict(True)
                self.up_0.append(wgts[0]["up"])
                self.up_1.append(wgts[1]["up"])
                self.pi_0.append(wgts[0]["pi"])
                self.ip_0.append(wgts[0]["ip"])
                self.down_0.append(wgts[0]["down"])


    def evaluate(self) -> bool:
        self.up_0 = np.array(self.up_0)
        self.pi_0 = np.array(self.pi_0)
        self.ip_0 = np.array(self.ip_0)
        self.down_0 = np.array(self.down_0)
        self.up_1 = np.array(self.up_1)
        return records_match(self.up_0.flatten(), self.numpy_net.weight_record[0]["up"].flatten()) \
            and records_match(self.pi_0.flatten(), self.numpy_net.weight_record[0]["pi"].flatten()) \
            and records_match(self.ip_0.flatten(), self.numpy_net.weight_record[0]["ip"].flatten()) \
            and records_match(self.up_1.flatten(), self.numpy_net.weight_record[-1]["up"].flatten()) \
            and records_match(self.down_0.flatten(), self.numpy_net.weight_record[0]["down"].flatten())

    def plot_results(self):

        fig, axes = plt.subplots(2, 3, sharex=True, sharey="col", constrained_layout=True)
        cmap = plt.cm.get_cmap('hsv', max(self.dims)+1)
        linestyles = ["solid", "dotted", "dashdot", "dashed"]

        for i, (name, layer) in enumerate(zip(["up", "down", "ip", "pi", "up"], [0, 0, 0, 0, 1])):

            weights_nest = eval(f"self.{name}_{layer}")
            weights_numpy = self.numpy_net.weight_record[layer][name]
            axes[i//3][i%3].set_title(name)
            for sender in range(weights_nest.shape[2]):
                for target in range(weights_nest.shape[1]):
                    col = cmap(sender)
                    style = linestyles[target]
                    axes[i//3][i%3].plot(weights_nest[:, target, sender], linestyle="solid", color=col)
                    axes[i//3][i%3].plot(weights_numpy[:, target, sender], linestyle="dashed", color=col, alpha= 0.8)

        axes[0][0].set_ylabel("NEST computed")
        axes[1][0].set_ylabel("Target activation")
