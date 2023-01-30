from test_utils import *
from tests01_neuron_dynamics import *


class PlasticityYH(DynamicsYH):

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)
        self.eta = 0.04

        conn = nest.GetConnections(self.neuron_01, self.neuron_02)
        if spiking_neurons:
            conn.eta = self.eta/(self.weight_scale**2 * 30)
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
        self.weight_ = []

        for T, amp in zip(self.sim_times, self.stim_amps):
            self.neuron_01.set({"soma": {"I_e": amp}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):
                delta_u_x = -self.g_l_eff * U_h + amp
                delta_u_h = -self.g_l_eff * U_y + V_by * self.g_d

                delta_tilde_w = -tilde_w + (r_y - self.phi(V_by * self.lambda_out)) * r_h

                U_h = U_h + self.delta_t * delta_u_x
                r_h = self.phi(U_h)

                V_by = r_h * self.weight
                U_y = U_y + self.delta_t * delta_u_h
                r_y = self.phi(U_y)

                tilde_w = tilde_w + (self.delta_t/self.tau_delta) * delta_tilde_w
                self.weight = self.weight + self.eta * self.delta_t * tilde_w
                self.weight_.append(self.weight)

                self.UH.append(U_h)
                self.UY.append(U_y)
                self.VBY.append(V_by)

    def evaluate(self) -> bool:
        weight_df = pd.DataFrame.from_dict(self.wr.events).drop_duplicates("times")
        if self.spiking_neurons:
            weight_df["weights"] *= self.weight_scale
        weight_df = weight_df.set_index("times")
        weight_df = weight_df.reindex(np.arange(0, self.SIM_TIME, self.delta_t))
        weight_df = weight_df.fillna(method="backfill").fillna(method="ffill")
        self.nest_weights = weight_df.weights.values

        return records_match(self.nest_weights, self.weight_)

    def plot_results(self):
        fig, axes = plt.subplots(1, 4, sharex=True, constrained_layout=True)

        axes[0].plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST")
        axes[0].plot(self.UH, label="target")
        axes[0].set_title("Input neuron Voltage")

        axes[1].plot(*zip(*read_multimeter(self.mm_02, "V_m.b")), label="NEST")
        axes[1].plot(self.VBY, label="target")
        axes[1].set_title("Hidden basal voltage")

        axes[2].plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST")
        axes[2].plot(self.UY, label="target")
        axes[2].set_title("Hidden somatic voltage")

        axes[3].plot(self.nest_weights, label="NEST")
        axes[3].plot(self.weight_, label="target")
        axes[3].legend()
        axes[3].set_title("synaptic weight")


class PlasticityHX(DynamicsHX):

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)
        self.eta = 0.04

        conn = nest.GetConnections(self.neuron_01, self.neuron_02)
        if spiking_neurons:
            conn.eta = self.eta/(self.weight_scale**2 * 30)
            conn.weight = self.weight / self.weight_scale
        else:
            conn.eta = self.eta

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
        self.weight_ = []

        for T, amp in zip(self.sim_times, self.stim_amps):
            self.neuron_01.set({"soma": {"I_e": amp/self.tau_x}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):
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
                self.weight_.append(self.weight)

                self.UX.append(U_x)
                self.UH.append(U_h)
                self.VBH.append(V_bh)

    def evaluate(self) -> bool:
        weight_df = pd.DataFrame.from_dict(self.wr.events).drop_duplicates("times")
        if self.spiking_neurons:
            weight_df["weights"] *= self.weight_scale
        weight_df = weight_df.set_index("times")
        weight_df = weight_df.reindex(np.arange(0, self.SIM_TIME, self.delta_t))
        weight_df = weight_df.fillna(method="backfill").fillna(method="ffill")
        self.nest_weights = weight_df.weights.values

        return records_match(self.nest_weights, self.weight_)

    def plot_results(self):
        fig, axes = plt.subplots(1, 4, sharex=True, constrained_layout=True)

        axes[0].plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST")
        axes[0].plot(self.UX, label="target")
        axes[0].set_title("Input neuron Voltage")

        axes[1].plot(*zip(*read_multimeter(self.mm_02, "V_m.b")), label="NEST")
        axes[1].plot(self.VBH, label="target")
        axes[1].set_title("Hidden basal voltage")

        axes[2].plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST")
        axes[2].plot(self.UH, label="target")
        axes[2].set_title("Hidden somatic voltage")

        axes[3].plot(self.nest_weights, label="NEST")
        axes[3].plot(self.weight_, label="target")
        axes[3].legend()
        axes[3].set_title("synaptic weight")


class PlasticityHI(DynamicsHI):
    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)
        self.eta = 0.04

        conn = nest.GetConnections(self.neuron_01, self.neuron_02)
        if spiking_neurons:
            conn.eta = self.eta/(self.weight_scale**2 * 30)
            conn.weight = self.weight / self.weight_scale
        else:
            conn.eta = self.eta

    def run(self):
        U_x = 0
        r_x = 0
        V_ah = 0
        U_h = 0
        r_h = 0
        delta_tilde_w = 0
        tilde_w = 0
        self.UX = []
        self.UH = []
        self.VBH = []
        self.weight_ = []

        for T, amp in zip(self.sim_times, self.stim_amps):
            self.neuron_01.set({"soma": {"I_e": amp/self.tau_x}})
            nest.Simulate(T)
            for i in range(int(T/self.delta_t)):

                delta_tilde_w = -tilde_w - V_ah * r_x

                delta_u_x = -U_x + amp
                U_x = U_x + (self.delta_t/self.tau_x) * delta_u_x
                r_x = self.phi(U_x)

                V_ah = r_x * self.weight

                delta_u_h = -self.g_l_eff * U_h + V_ah * self.g_a
                U_h = U_h + self.delta_t * delta_u_h
                r_h = self.phi(U_h)

                tilde_w = tilde_w + (self.delta_t/self.tau_delta) * delta_tilde_w
                self.weight = self.weight + self.eta * self.delta_t * tilde_w
                self.weight_.append(self.weight)

                self.UX.append(U_x)
                self.UH.append(U_h)
                self.VBH.append(V_ah)

    def evaluate(self) -> bool:
        weight_df = pd.DataFrame.from_dict(self.wr.events).drop_duplicates("times")
        weight_df["weights"] *= self.weight_scale
        weight_df = weight_df.set_index("times")
        weight_df = weight_df.reindex(np.arange(0, self.SIM_TIME, self.delta_t))
        weight_df = weight_df.fillna(method="backfill").fillna(method="ffill")
        self.nest_weights = weight_df.weights.values

        return records_match(self.nest_weights, self.weight_)

    def plot_results(self):
        fig, axes = plt.subplots(1, 4, sharex=True, constrained_layout=True)

        axes[0].plot(*zip(*read_multimeter(self.mm_01, "V_m.s")), label="NEST")
        axes[0].plot(self.UX, label="target")
        axes[0].set_title("Input neuron Voltage")

        axes[1].plot(*zip(*read_multimeter(self.mm_02, "V_m.a_lat")), label="NEST")
        axes[1].plot(self.VBH, label="target")
        axes[1].set_title("Hidden apical voltage")

        axes[2].plot(*zip(*read_multimeter(self.mm_02, "V_m.s")), label="NEST")
        axes[2].plot(self.UH, label="target")
        axes[2].set_title("Hidden somatic voltage")

        axes[3].plot(self.nest_weights, label="NEST")
        axes[3].plot(self.weight_, label="target")
        axes[3].legend()
        axes[3].set_title("synaptic weight")


class NetworkPlasticity(TestClass):
    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        sim["teacher"] = False
        sim["noise"] = False
        sim["dims"] = [4, 3, 2]

        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)
        self.numpy_net = NumpyNetwork(deepcopy(sim), deepcopy(nrn), deepcopy(syn))
        if spiking_neurons:
            nrn["input"]["gamma"] = self.weight_scale
            nrn["pyr"]["gamma"] = self.weight_scale * nrn["pyr"]["gamma"]
            nrn["intn"]["gamma"] = self.weight_scale * nrn["intn"]["gamma"]
            syn["wmin_init"] = -1/self.weight_scale
            syn["wmax_init"] = 1/self.weight_scale
            for syn_name in ["hx", "yh", "hy", "hi", "ih"]:
                if "eta" in syn[syn_name]:
                    syn[syn_name]["eta"] /= self.weight_scale**2 * 30
        else:
            self.weight_scale = 1


        self.nest_net = NestNetwork(sim, nrn, syn)

    def run(self):
        input_currents = np.random.random(self.dims[0])
        target_currents = np.random.random(self.dims[2])

        self.nest_net.set_input(input_currents)
        self.nest_net.set_target(target_currents)

        self.numpy_net.set_input(input_currents)
        self.numpy_net.set_target(target_currents)

        weights = self.nest_net.get_weight_dict()

        for conn in ["hi", "ih", "hx", "hy", "yh"]:
            self.numpy_net.conns[conn]["w"] = weights[conn] * self.weight_scale

        self.sim_time = 100
        self.nest_net.simulate(self.sim_time)
        for i in range(int(self.sim_time/self.delta_t)):
            self.numpy_net.simulate(self.numpy_net.train_static)

    def evaluate(self) -> bool:
        weight_df = pd.DataFrame.from_dict(self.wr.events)
        weight_df["weights"] *= self.weight_scale
        weight_df = weight_df.groupby(["senders", "targets"])

        self.hx_nest = utils.read_wr(
            weight_df, self.nest_net.pyr_pops[0], self.nest_net.pyr_pops[1], self.sim_time, self.delta_t)
        self.hi_nest = utils.read_wr(
            weight_df, self.nest_net.intn_pops[0], self.nest_net.pyr_pops[1], self.sim_time, self.delta_t)
        self.ih_nest = utils.read_wr(
            weight_df, self.nest_net.pyr_pops[1], self.nest_net.intn_pops[0],  self.sim_time, self.delta_t)
        self.hy_nest = utils.read_wr(
            weight_df, self.nest_net.pyr_pops[2], self.nest_net.pyr_pops[1], self.sim_time, self.delta_t)
        self.yh_nest = utils.read_wr(
            weight_df, self.nest_net.pyr_pops[1], self.nest_net.pyr_pops[2], self.sim_time, self.delta_t)

        return records_match(self.hx_nest.flatten(), self.numpy_net.conns["hx"]["record"].flatten()) \
            and records_match(self.hi_nest.flatten(), self.numpy_net.conns["hi"]["record"].flatten()) \
            and records_match(self.ih_nest.flatten(), self.numpy_net.conns["ih"]["record"].flatten()) \
            and records_match(self.yh_nest.flatten(), self.numpy_net.conns["yh"]["record"].flatten()) \
            and records_match(self.hy_nest.flatten(), self.numpy_net.conns["hy"]["record"].flatten())

    def plot_results(self):
        fig, axes = plt.subplots(2, 5, sharex=True, sharey="col", constrained_layout=True)
        cmap = plt.cm.get_cmap('hsv', max(self.dims)+1)
        linestyles = ["solid", "dotted", "dashdot", "dashed"]

        for i, name in enumerate(["hx", "yh", "ih", "hi", "hy"]):

            weights_nest = eval(f"self.{name}_nest")
            weights_numpy = self.numpy_net.conns[name]["record"]
            axes[0][i].set_title(name)
            for sender in range(weights_nest.shape[2]):
                for target in range(weights_nest.shape[1]):
                    col = cmap(sender)
                    style = linestyles[target]
                    axes[0][i].plot(weights_nest[:, target, sender], linestyle=style, color=col)
                    axes[1][i].plot(weights_numpy[:, target, sender], linestyle=style, color=col)

        axes[0][0].set_ylabel("NEST computed")
        axes[1][0].set_ylabel("Target activation")


class Dummy(TestClass):

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        super().__init__(nrn, sim, syn, spiking_neurons, **kwargs)

    def evaluate(self) -> bool:
        pass

    def plot_results(self):
        pass
