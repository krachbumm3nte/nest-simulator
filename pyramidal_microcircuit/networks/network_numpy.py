import numpy as np
from sklearn.metrics import mean_squared_error as mse
from time import time
from copy import deepcopy

# These values reappear over and over again in the computation. Writing self.value 5 times per line bloats
# the simulate() function to horrific degrees. Since these values do not change at simulation time, They are being
# set as global variables once in the constructor to be called from within the class. This solution is pretty cursed,
# but the oop solution sadly makes the code completely unreadable.
g_a, g_d, g_l, g_s, g_si, tau_x, tau_delta, noise_factor, delta_t = [0 for i in range(9)]
phi, phi_inverse = None, None


class NumpyNetwork:

    def __init__(self, sim, nrn, syns) -> None:
        self.sim = deepcopy(sim)  # simulation parameters
        self.nrn = deepcopy(nrn)  # neuron parameters
        self.syns = deepcopy(syns)  # synapse parameters

        self.dims = sim["dims"]
        self.conns = {}
        self.phi = nrn["phi"]
        self.phi_inverse = nrn["phi_inverse"]
        self.record_voltages = True

        # Oh lord forgive me
        global g_a, g_d, g_l, g_s, g_si, noise_factor, delta_t, tau_x, tau_delta, phi, phi_inverse
        g_a = nrn["g_a"]
        g_d = nrn["g_d"]
        g_l = nrn["g_l"]
        g_s = nrn["g_s"]
        g_si = nrn["g_si"]
        tau_x = nrn["tau_x"]
        noise_factor = sim["noise_factor"] if sim["noise"] else 0
        tau_delta = syns["tau_Delta"]
        delta_t = sim["delta_t"]
        phi = nrn["phi"]
        phi_inverse = nrn["phi_inverse"]

        self.U_x_record = np.asmatrix(np.zeros((0, self.dims[0])))
        self.U_h_record = np.asmatrix(np.zeros((0, self.dims[1])))
        self.V_ah_record = np.asmatrix(np.zeros((0, self.dims[1])))
        self.V_bh_record = np.asmatrix(np.zeros((0, self.dims[1])))
        self.U_i_record = np.asmatrix(np.zeros((0, self.dims[2])))
        self.V_bi_record = np.asmatrix(np.zeros((0, self.dims[2])))
        self.U_y_record = np.asmatrix(np.zeros((0, self.dims[2])))
        self.V_by_record = np.asmatrix(np.zeros((0, self.dims[2])))
        self.output_loss = []

        self.setup_populations(self.syns, self.nrn)

        self.iteration = 0

    def gen_weights(self, lr, next_lr):
        return np.random.uniform(self.syns["wmin_init"], self.syns["wmax_init"], (next_lr, lr))

    def setup_populations(self, syns, nrn):
        self.U_x = np.asmatrix(np.zeros(self.dims[0]))

        self.U_h = np.asmatrix(np.zeros(self.dims[1]))
        self.V_bh = np.asmatrix(np.zeros(self.dims[1]))
        self.V_ah = np.asmatrix(np.zeros(self.dims[1]))
        self.r_h = self.phi(self.U_h)

        self.U_i = np.asmatrix(np.zeros(self.dims[2]))
        self.V_bi = np.asmatrix(np.zeros(self.dims[2]))
        self.r_i = self.phi(self.U_i)

        self.U_y = np.asmatrix(np.zeros(self.dims[2]))
        self.V_by = np.asmatrix(np.zeros(self.dims[2]))
        self.r_y = self.phi(self.U_y)
        self.y = np.asmatrix(np.random.random(self.dims[2]))

        self.conn_names = ["hx", "yh", "ih", "hi", "hy"]

        conn_setup = {
            "hx": {"eta": syns["hx"]["eta"], "in": self.dims[0], "out": self.dims[1], "init_scale": 1},  # 0.1},
            "yh": {"eta": syns["yh"]["eta"], "in": self.dims[1], "out": self.dims[2], "init_scale": 1},  # 0.1},
            "ih": {"eta": syns["ih"]["eta"], "in": self.dims[1], "out": self.dims[2], "init_scale": 1},  # 0.1},
            "hi": {"eta": syns["hi"]["eta"], "in": self.dims[2], "out": self.dims[1], "init_scale": 1},  # 0.1},
            "hy": {"eta": syns["hy"]["eta"], "in": self.dims[2], "out": self.dims[1], "init_scale": 1},  # 1/gamma}
        }

        for name, p in conn_setup.items():
            self.conns[name] = {
                "eta": p["eta"],
                "w": np.asmatrix(np.random.random((p["out"], p["in"])) * 2 - 1) * p["init_scale"],
                "dt_w": np.asmatrix(np.zeros((p["out"], p["in"]))),
                "t_w": np.asmatrix(np.zeros((p["out"], p["in"]))),
                "record": np.zeros((0, p["out"], p["in"]))
            }

        self.hx_teacher = np.asmatrix(np.random.random((self.dims[1], self.dims[0])) * 2 - 1)
        self.yh_teacher = np.asmatrix(np.random.random((self.dims[2], self.dims[1])) * 2 - 1) / nrn["gamma"]

    def train(self, T):

        for i in range(int(T/delta_t)):
            self.simulate(self.train_nothing)

    def test(self, T):
        for i in range(int(T/delta_t)):
            self.simulate(self.train_nothing)  # do not inject output layer current during testing
            self.output_pred = phi((g_d / (g_d + g_l)) +
                                   self.conns["yh"]["w"] * phi((g_d / (g_d + g_l + g_a)) * self.V_bh).T).T

            self.output_loss.append(mse(np.asarray(self.output_pred), np.asarray(self.y)))

    def train_match_teacher(self):
        y_teacher = phi(self.yh_teacher * phi(self.hx_teacher * self.U_x.T)).T
        return phi_inverse(y_teacher)

    def train_inverse(self):
        assert self.dims[0] == self.dims[-1]
        return -self.U_x

    def train_nothing(self):
        return np.asmatrix(np.zeros(self.dims[2]))

    def train_static(self):
        return np.asmatrix(np.full(self.dims[2], self.target_amp))

    def simulate(self, train_function):

        # start = time()
        # print("timing...")

        delta_u_x = -self.U_x + self.I_x
        delta_u_h = -(g_l + g_d + g_a) * self.U_h + g_d * self.V_bh + g_a * self.V_ah
        delta_u_y = -(g_l + g_d + g_a) * self.U_y + g_d * self.V_by + g_s * self.y
        delta_u_i = -(g_l + g_d + g_a) * self.U_i + g_d * self.V_bi + g_si * self.U_y

        # stop = time()
        # print(stop - start)
        # start = time()

        self.conns["hx"]["dt_w"] = -self.conns["hx"]["t_w"] + \
            np.outer(self.r_h - phi((g_d * self.V_bh)/(g_l + g_d + g_a)), self.U_x)

        self.conns["yh"]["dt_w"] = -self.conns["yh"]["t_w"] + \
            np.outer(self.r_y - phi((g_d * self.V_by)/(g_l + g_d)), self.r_h)

        # output to hidden synapses learn this way, but eta is almost always zero, so this saves some compute time
        # self.conns["hy"]["dt_w"] = -self.conns["hy"]["t_w"] + \
        #     np.outer(self.r_h - phi(self.conns["hy"]["w"] * self.r_y.T).T, self.r_y)

        self.conns["ih"]["dt_w"] = -self.conns["ih"]["t_w"] + \
            np.outer(self.r_i - phi((g_d * self.V_bi)/(g_l + g_d)), self.r_h)

        self.conns["hi"]["dt_w"] = np.subtract(np.outer(-self.V_ah, self.r_i), self.conns["hi"]["t_w"])

        # stop = time()
        # print(stop - start)
        # start = time()

        self.U_x = self.U_x + (delta_t/tau_x) * delta_u_x

        self.y = train_function()

        self.V_bh = (self.conns["hx"]["w"] * self.U_x.T).T  # Note that input neurons do not use a transfer function
        self.V_ah = (self.conns["hy"]["w"] * self.r_y.T + self.conns["hi"]["w"] * self.r_i.T).T
        self.U_h = self.U_h + delta_t * delta_u_h + noise_factor * np.random.standard_normal(self.dims[1])
        self.r_h = phi(self.U_h)

        self.V_by = (self.conns["yh"]["w"] * self.r_h.T).T
        self.U_y = self.U_y + delta_t * delta_u_y + noise_factor * np.random.standard_normal(self.dims[2])
        self.r_y = phi(self.U_y)

        self.V_bi = (self.conns["ih"]["w"] * self.r_h.T).T
        self.U_i = self.U_i + delta_t * delta_u_i + noise_factor * np.random.standard_normal(self.dims[2])
        self.r_i = phi(self.U_i)

        # stop = time()
        # print(stop - start)
        # start = time()

        store_state = self.record_voltages and self.iteration % int(self.sim["record_interval"]/delta_t) == 0

        for name, d in self.conns.items():
            d["t_w"] = d["t_w"] + (delta_t/tau_delta) * d["dt_w"]
            d["w"] = d["w"] + d["eta"] * delta_t * d["t_w"]
            if store_state:
                d["record"] = np.append(d["record"], np.expand_dims(d["w"], axis=0), axis=0)
            # if name == "ih":
            #     print(f"syn2: {d['w'][0,0]:.5f}, {d['t_w'][0,0]:.5f}, {d['dt_w'][0,0]:.5f}")

        # stop = time()
        # print(f"{stop - start}")
        # start = time()
        if store_state:
            self.U_x_record = np.append(self.U_x_record, self.U_x, axis=0)
            self.U_h_record = np.append(self.U_h_record, self.U_h, axis=0)
            self.V_ah_record = np.append(self.V_ah_record, self.V_ah, axis=0)
            self.V_bh_record = np.append(self.V_bh_record, self.V_bh, axis=0)
            self.U_i_record = np.append(self.U_i_record, self.U_i, axis=0)
            self.V_bi_record = np.append(self.V_bi_record, self.V_bi, axis=0)
            self.U_y_record = np.append(self.U_y_record, self.U_y, axis=0)
            self.V_by_record = np.append(self.V_by_record, self.V_by, axis=0)

            self.output_pred = self.phi((g_d / (g_d + g_l)) + self.conns["yh"]
                                        ["w"] * self.phi((g_d / (g_d + g_l + g_a)) * self.V_bh).T).T
            self.output_loss.append(mse(np.asarray(self.output_pred), np.asarray(self.y)))

        self.iteration += 1

        # stop = time()
        # print(stop - start)
        # start = time()
        # print("done")

    def set_input(self, input_currents):
        """Inject a constant current into all neurons in the input layer.

        Arguments:
            input_currents -- Iterable of length equal to the input dimension.
        """
        self.I_x = input_currents
