import numpy as np
from sklearn.metrics import mean_squared_error as mse
from time import time
from .network import Network
import torch
import torch.nn as nn
import utils
from copy import deepcopy

# These values reappear over and over again in the computation. Writing self.value 5 times per line bloats
# the simulate() function to horrific degrees. Since these values do not change at simulation time, They are being
# set as global variables once in the constructor to be called from within the class. This solution is pretty cursed,
# but the oop solution sadly makes the code completely unreadable.
g_a, g_d, g_l, g_s, g_si, tau_x, tau_delta, noise_factor, delta_t, leakage, f_1, f_2 = [0 for i in range(12)]


torch.autograd.profiler.profile(enabled=False)


class TorchNetwork(Network):

    def __init__(self, sim, nrn, syns) -> None:
        super().__init__(sim, nrn, syns)

        self.conns = {}
        self.record_voltages = True

        # Oh lord forgive me
        global g_a, g_d, g_l, g_s, g_si, noise_factor, delta_t, tau_x, tau_delta, leakage, f_1, f_2
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
        f_1 = g_d / (g_l + g_d)
        f_2 = g_d / (g_l + g_a + g_d)

        self.U_x_record = torch.as_tensor(utils.zeros((0, self.dims[0])))
        self.U_h_record = torch.as_tensor(utils.zeros((0, self.dims[1])))
        self.V_ah_record = torch.as_tensor(utils.zeros((0, self.dims[1])))
        self.V_bh_record = torch.as_tensor(utils.zeros((0, self.dims[1])))
        self.U_i_record = torch.as_tensor(utils.zeros((0, self.dims[2])))
        self.V_bi_record = torch.as_tensor(utils.zeros((0, self.dims[2])))
        self.U_y_record = torch.as_tensor(utils.zeros((0, self.dims[2])))
        self.V_by_record = torch.as_tensor(utils.zeros((0, self.dims[2])))
        self.output_loss = []

        self.setup_populations(self.syns, self.nrn)

        self.iteration = 0

    def setup_populations(self, syns, nrn):
        self.U_x = torch.tensor(utils.zeros(self.dims[0]))
        self.U_h = torch.tensor(utils.zeros(self.dims[1]))
        self.V_bh = torch.tensor(utils.zeros(self.dims[1]))
        self.V_ah = torch.tensor(utils.zeros(self.dims[1]))
        self.U_i = torch.tensor(utils.zeros(self.dims[2]))
        self.V_bi = torch.tensor(utils.zeros(self.dims[2]))
        self.U_y = torch.tensor(utils.zeros(self.dims[2]))
        self.V_by = torch.tensor(utils.zeros(self.dims[2]))
        self.y = torch.rand(self.dims[2])

        self.r_h = self.phi(self.U_h)
        self.r_i = self.phi(self.U_i)
        self.r_y = self.phi(self.U_y)
        self.conn_names = ["hx", "yh", "ih", "hi", "hy"]

        conn_setup = {
            "hx": {"in": self.dims[0], "out": self.dims[1], "init_scale": 0.1},
            "yh": {"in": self.dims[1], "out": self.dims[2], "init_scale": 0.1},
            "ih": {"in": self.dims[1], "out": self.dims[2], "init_scale": 0.1},
            "hi": {"in": self.dims[2], "out": self.dims[1], "init_scale": 0.1},
            "hy": {"in": self.dims[2], "out": self.dims[1], "init_scale": 1/nrn["gamma"]}
        }

        for name, p in conn_setup.items():
            self.conns[name] = {
                "eta": syns[name]["eta"],
                "w": nn.Linear(p["in"], p["out"], bias=False),
                "dt_w": torch.tensor(utils.zeros((p["out"], p["in"]))),
                "t_w": torch.tensor(utils.zeros((p["out"], p["in"]))),
                "record": utils.zeros((0, p["out"], p["in"]))
            }
            nn.init.uniform_(self.conns[name]["w"].weight, -p["init_scale"], p["init_scale"])

        if self.teacher:
            self.hx_teacher = nn.Linear(self.dims[0], self.dims[1], False)
            self.yh_teacher = nn.Linear(self.dims[1], self.dims[2], False)
            nn.init.uniform_(self.hx_teacher.weight, -1, 1)
            nn.init.uniform_(self.yh_teacher.weight, -1/self.nrn["gamma"], 1/self.nrn["gamma"])

    def phi(self, x):
        return self.gamma * torch.log(1 + torch.exp(self.beta * (x - self.theta)))

    def phi_inverse(self, x):
        return (1 / self.beta) * (self.beta * self.theta + torch.log(torch.exp(x/self.gamma) - 1))

    def train(self, input_currents, T):

        self.set_input(input_currents)

        for i in range(int(T/delta_t)):
            self.simulate(self.train_match_teacher)

    def test(self, T):
        for i in range(int(T/delta_t)):
            # do not inject output layer current during testing
            self.simulate(self.train_match_teacher if self.teacher else self.train_nothing)
            self.output_pred = self.phi(f_1 * self.conns["yh"]["w"](self.phi(f_2 * self.V_bh)))

            self.output_loss.append(mse(np.asarray(self.y), np.asarray(self.output_pred)))

    def train_match_teacher(self):
        return self.phi(self.yh_teacher(self.phi(self.hx_teacher(self.U_x))))

    def train_inverse(self):
        assert self.dims[0] == self.dims[-1]
        return -self.U_x

    def train_nothing(self):
        return torch.tensor(utils.zeros(self.dims[2]))

    def train_static(self):
        return torch.tensor(np.full(self.dims[2], self.target_amp))

    def simulate(self, train_function):

        start = time()
        print("timing...")

        delta_u_x = -self.U_x + self.I_x
        delta_u_h = -leakage * self.U_h + g_d * self.V_bh + g_a * self.V_ah
        delta_u_y = -leakage * self.U_y + g_d * self.V_by + g_s * self.y
        delta_u_i = -leakage * self.U_i + g_d * self.V_bi + g_si * self.U_y

        stop = time()
        print(stop - start)
        start = time()

        self.conns["hx"]["dt_w"] = -self.conns["hx"]["t_w"] + \
            torch.outer(self.r_h - self.phi(self.V_bh * f_2), self.U_x)

        self.conns["yh"]["dt_w"] = -self.conns["yh"]["t_w"] + \
            torch.outer(self.r_y - self.phi(self.V_by * f_1), self.r_h)

        # output to hidden synapses learn this way, but eta is almost always zero, so this saves some compute time
        # self.conns["hy"]["dt_w"] = -self.conns["hy"]["t_w"] + \
        #     np.outer(self.r_h - self.phi(self.conns["hy"]["w"] * self.r_y.T).T, self.r_y)

        self.conns["ih"]["dt_w"] = -self.conns["ih"]["t_w"] + \
            torch.outer(self.r_i - self.phi(self.V_bi * f_1), self.r_h)

        self.conns["hi"]["dt_w"] = torch.subtract(torch.outer(-self.V_ah, self.r_i), self.conns["hi"]["t_w"])

        stop = time()
        print(stop - start)
        start = time()

        self.U_x = self.U_x + (delta_t/tau_x) * delta_u_x

        self.y = train_function()

        self.V_bh = self.conns["hx"]["w"](self.U_x)  # Note that input neurons do not use a transfer function
        self.V_ah = self.conns["hy"]["w"](self.r_y) + self.conns["hi"]["w"](self.r_i)
        self.U_h = self.U_h + delta_t * delta_u_h + noise_factor * torch.rand(self.dims[1]) * 2 - 1
        self.r_h = self.phi(self.U_h)

        self.V_by = self.conns["yh"]["w"](self.r_h)
        self.U_y = self.U_y + delta_t * delta_u_y + noise_factor * torch.rand(self.dims[2]) * 2 - 1
        self.r_y = self.phi(self.U_y)

        self.V_bi = self.conns["ih"]["w"](self.r_h)
        self.U_i = self.U_i + delta_t * delta_u_i + noise_factor * torch.rand(self.dims[2]) * 2 - 1
        self.r_i = self.phi(self.U_i)

        stop = time()
        print(stop - start)
        start = time()

        store_state = self.record_voltages and self.iteration % int(self.sim["record_interval"]/delta_t) == 0

        for name, d in self.conns.items():
            d["t_w"] += (delta_t/tau_delta) * d["dt_w"]
            d["w"].weight += d["eta"] * delta_t * d["t_w"]
            if store_state:
                d["record"] = np.append(d["record"], np.expand_dims(deepcopy(d["w"]).cpu().weight, axis=0), axis=0)

        stop = time()
        print(f"{stop - start}")
        start = time()
        if store_state:
            self.U_x_record = torch.cat((self.U_x_record, self.U_x.unsqueeze(dim=0)), axis=0)
            self.U_h_record = torch.cat((self.U_h_record, self.U_h.unsqueeze(dim=0)), axis=0)
            self.V_ah_record = torch.cat((self.V_ah_record, self.V_ah.unsqueeze(dim=0)), axis=0)
            self.V_bh_record = torch.cat((self.V_bh_record, self.V_bh.unsqueeze(dim=0)), axis=0)
            self.U_i_record = torch.cat((self.U_i_record, self.U_i.unsqueeze(dim=0)), axis=0)
            self.V_bi_record = torch.cat((self.V_bi_record, self.V_bi.unsqueeze(dim=0)), axis=0)
            self.U_y_record = torch.cat((self.U_y_record, self.U_y.unsqueeze(dim=0)), axis=0)
            self.V_by_record = torch.cat((self.V_by_record, self.V_by.unsqueeze(dim=0)), axis=0)

            self.output_pred = self.phi(f_1 * self.conns["yh"]["w"](self.phi(f_2 * self.V_bh)))
            self.output_loss.append(mse(self.y.cpu(), self.output_pred.cpu()))

        self.iteration += 1

        stop = time()
        print(stop - start)
        start = time()
        print("done")

    def set_input(self, input_currents):
        """Inject a constant current into all neurons in the input layer.

        Arguments:
            input_currents -- Iterable of length equal to the input dimension.
        """
        self.I_x = torch.as_tensor(input_currents, dtype=torch.float32)
