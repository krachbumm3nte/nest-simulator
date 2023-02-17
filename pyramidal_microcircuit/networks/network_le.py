import numpy as np
from sklearn.metrics import mean_squared_error as mse
from time import time
from .network import Network

# These values reappear over and over again in the computation. Writing self.value 5 times per line bloats
# the simulate() function to horrific degrees. Since these values do not change at simulation time, They are being
# set as global variables once in the constructor to be called from within the class. This solution is pretty cursed,
# but the oop solution sadly makes the code completely unreadable.
g_a, g_d, g_l, g_s, g_si, tau_x, tau_delta, noise_factor, delta_t, leakage, lambda_out, lambda_ah, lambda_bh = [
    0 for i in range(13)]


class LatentEquilibriumNetwork(Network):

    def __init__(self, sim, nrn, syns) -> None:
        super().__init__(sim, nrn, syns)

        self.conns = {}
        self.record_voltages = True

        # Oh lord forgive me
        global g_a, g_d, g_l, g_s, g_si, noise_factor, delta_t, tau_x, tau_delta, leakage, lambda_out, lambda_ah, lambda_bh
        g_a = nrn["g_a"]
        g_d = nrn["g_d"]
        g_l = nrn["g_l"]
        g_s = nrn["g_s"]
        g_si = nrn["g_si"]
        tau_x = nrn["tau_x"]
        noise_factor = sim["noise_factor"] if sim["noise"] else 0
        tau_delta = syns["tau_Delta"]
        delta_t = sim["delta_t"]

        leakage = g_l + g_a + g_d
        lambda_out = nrn["lambda_out"]
        lambda_ah = nrn["lambda_ah"]
        lambda_bh = nrn["lambda_bh"]

        self.U_x_record = np.zeros((1, self.dims[0]))
        self.U_h_record = np.zeros((1, self.dims[1]))
        self.V_ah_record = np.zeros((1, self.dims[1]))
        self.V_bh_record = np.zeros((1, self.dims[1]))
        self.U_i_record = np.zeros((1, self.dims[2]))
        self.V_bi_record = np.zeros((1, self.dims[2]))
        self.U_y_record = np.zeros((1, self.dims[2]))
        self.V_by_record = np.zeros((1, self.dims[2]))
        self.output_loss = []

        self.setup_populations(self.syns, self.nrn)

        self.iteration = 0

    def setup_populations(self, syns, nrn):
        self.U_x = np.zeros(self.dims[0])
        self.U_h = np.zeros(self.dims[1])
        self.V_bh = np.zeros(self.dims[1])
        self.V_ah = np.zeros(self.dims[1])
        self.U_i = np.zeros(self.dims[2])
        self.V_bi = np.zeros(self.dims[2])
        self.U_y = np.zeros(self.dims[2])
        self.V_by = np.zeros(self.dims[2])
        self.y = np.random.random(self.dims[2])

        self.r_h = self.phi(self.U_h)
        self.r_i = self.phi(self.U_i)
        self.r_y = self.phi(self.U_y)
        self.conn_names = ["hx", "yh", "ih", "hi", "hy"]

        conn_setup = {
            "hx": {"in": self.dims[0], "out": self.dims[1], "init_scale": syns["w_init_hx"]},
            "yh": {"in": self.dims[1], "out": self.dims[2], "init_scale": syns["w_init_yh"]},
            "ih": {"in": self.dims[1], "out": self.dims[2], "init_scale": syns["w_init_ih"]},
            "hi": {"in": self.dims[2], "out": self.dims[1], "init_scale": syns["w_init_hi"]},
            "hy": {"in": self.dims[2], "out": self.dims[1], "init_scale": syns["w_init_hy"]}
        }

        for name, p in conn_setup.items():
            self.conns[name] = {
                "eta": syns[name]["eta"] if "eta" in syns[name] else 0,
                "w": self.gen_weights(p["in"], p["out"], -p["init_scale"], p["init_scale"]),
                "dt_w": np.zeros((p["out"], p["in"])),
                "t_w": np.zeros((p["out"], p["in"])),
                "record": np.zeros((0, p["out"], p["in"]))
            }

    def train(self, input_currents, T):

        self.set_input(input_currents)

        for i in range(int(T/delta_t)):
            self.simulate(self.target_teacher if self.teacher else self.target_nothing)

    def train_bars(self, T):

        inputs, target = self.generate_bar_data()

        self.set_input(inputs.flatten())
        target_curr = np.zeros(self.dims[-1])
        target_curr[target] = 1
        self.set_target(target_curr)
        for i in range(int(T/delta_t)):
            self.simulate(lambda: target_curr)

    def test(self, T):
        for i in range(int(T/delta_t)):
            # do not inject output layer current during testing
            self.simulate(self.target_teacher if self.teacher else self.target_nothing)
            self.output_pred = lambda_out * self.conns["yh"]["w"] @ self.phi(lambda_bh * self.V_bh)

            self.output_loss.append(mse(self.y, self.output_pred))

    def target_teacher(self):
        return self.wyh_trgt @ self.phi(self.whx_trgt @ self.U_x)

    def target_invert(self):
        assert self.dims[0] == self.dims[-1]
        return -self.U_x

    def target_nothing(self):
        return np.zeros(self.dims[2])

    def train_static(self):
        return self.target_currents

    def simulate(self, train_function):

        delta_u_x = -self.U_x + self.I_x
        delta_u_h = -leakage * self.U_h + g_d * self.V_bh + g_a * self.V_ah
        delta_u_y = -leakage * self.U_y + g_d * self.V_by + g_s * self.y
        delta_u_i = -leakage * self.U_i + g_d * self.V_bi + g_si * self.U_y

        self.conns["hx"]["dt_w"] = -self.conns["hx"]["t_w"] + \
            np.outer(self.r_h - self.phi(self.V_bh * lambda_bh), self.U_x)

        self.conns["yh"]["dt_w"] = -self.conns["yh"]["t_w"] + \
            np.outer(self.r_y - self.phi(self.V_by * lambda_out), self.r_h)

        # output to hidden synapses learn this way, but eta is almost always zero, so this saves some compute time
        # self.conns["hy"]["dt_w"] = -self.conns["hy"]["t_w"] + \
        #     np.outer(self.r_h - self.phi(self.conns["hy"]["w"] * self.r_y.T).T, self.r_y)

        self.conns["ih"]["dt_w"] = -self.conns["ih"]["t_w"] + \
            np.outer(self.r_i - self.phi(self.V_bi * lambda_out), self.r_h)

        self.conns["hi"]["dt_w"] = -self.conns["hi"]["t_w"] + np.outer(-self.V_ah, self.r_i)

        self.U_x = self.U_x + (delta_t/tau_x) * delta_u_x

        self.y = train_function()

        self.V_bh = self.conns["hx"]["w"] @ self.U_x  # Note that input neurons do not use a transfer function
        self.V_ah = self.conns["hy"]["w"] @ self.r_y + self.conns["hi"]["w"] @ self.r_i
        self.U_h = self.U_h + delta_t * delta_u_h  # + noise_factor * np.random.standard_normal(self.dims[1])
        self.r_h = self.phi(self.U_h)

        self.V_by = self.conns["yh"]["w"] @ self.r_h
        self.U_y = self.U_y + delta_t * delta_u_y  # + noise_factor * np.random.standard_normal(self.dims[2])
        self.r_y = self.phi(self.U_y)

        self.V_bi = self.conns["ih"]["w"] @ self.r_h
        self.U_i = self.U_i + delta_t * delta_u_i  # + noise_factor * np.random.standard_normal(self.dims[2])
        self.r_i = self.phi(self.U_i)

        store_state = self.record_voltages and self.iteration % int(self.sim["record_interval"]/delta_t) == 0

        for name, d in self.conns.items():
            d["t_w"] = d["t_w"] + (delta_t/tau_delta) * d["dt_w"]
            d["w"] = d["w"] + d["eta"] * delta_t * d["t_w"]
            if store_state:
                d["record"] = np.concatenate((d["record"], np.expand_dims(d["w"], axis=0)), axis=0)

        if store_state:
            self.U_x_record = np.concatenate((self.U_x_record, np.expand_dims(self.U_x, 0)), axis=0)
            self.U_h_record = np.concatenate((self.U_h_record, np.expand_dims(self.U_h, 0)), axis=0)
            self.V_ah_record = np.concatenate((self.V_ah_record, np.expand_dims(self.V_ah, 0)), axis=0)
            self.V_bh_record = np.concatenate((self.V_bh_record, np.expand_dims(self.V_bh, 0)), axis=0)
            self.U_i_record = np.concatenate((self.U_i_record, np.expand_dims(self.U_i, 0)), axis=0)
            self.V_bi_record = np.concatenate((self.V_bi_record, np.expand_dims(self.V_bi, 0)), axis=0)
            self.U_y_record = np.concatenate((self.U_y_record, np.expand_dims(self.U_y, 0)), axis=0)
            self.V_by_record = np.concatenate((self.V_by_record, np.expand_dims(self.V_by, 0)), axis=0)

            self.output_pred = self.phi(lambda_out * self.conns["yh"]["w"] @ self.phi(lambda_bh * self.V_bh))
            self.output_loss.append(mse(np.asarray(self.y), np.asarray(self.output_pred)))

        self.iteration += 1

    def set_input(self, input_currents):
        """Inject a constant current into all neurons in the input layer.

        Arguments:
            input_currents -- Iterable of length equal to the input dimension.
        """
        self.I_x = input_currents

    def get_weight_dict(self):
        weights = {}
        for k, v in self.conns.items():
            weights[k] = v["w"]
        return weights

    def set_target(self, target_currents):
        self.target_currents = target_currents