import nest
from params_rate import *
import numpy as np


class MathematicaNetwork:

    def __init__(self) -> None:
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
        self.V_ah_record = []
        self.U_i_record = []
        self.U_y_record = []
        self.setup_populations(init_self_pred)

    def gen_weights(self, lr, next_lr, w_min=wmin_init, w_max=wmax_init):
        return np.random.uniform(w_min, w_max, (next_lr, lr))

    def setup_populations(self, self_predicting):
        self.r_x = np.asmatrix(np.zeros(dims[0]))
        self.U_h = np.asmatrix(np.zeros(dims[1]))
        self.U_x = np.asmatrix(np.zeros(dims[0]))
        self.V_bh = np.asmatrix(np.zeros(dims[1]))
        self.V_ah = np.asmatrix(np.zeros(dims[1]))
        self.r_h = np.asmatrix(np.zeros(dims[1]))
        self.U_i = np.asmatrix(np.zeros(dims[2]))
        self.V_bi = np.asmatrix(np.zeros(dims[2]))
        self.r_i = np.asmatrix(np.zeros(dims[2]))
        self.U_y = np.asmatrix(np.zeros(dims[2]))
        self.V_by = np.asmatrix(np.zeros(dims[2]))
        self.r_y = np.asmatrix(np.zeros(dims[2]))

        self.conn_names = ["hx", "yh", "ih", "hi", "hy"]

        self.conns = {n: {"t_w": 0, "dt_w": 0, "record": [], "eta": 0, "w": 1} for n in self.conn_names}

        self.conns["hi"]["eta"] = eta_hi
        self.conns["ih"]["eta"] = eta_ih
        self.conns["hx"]["eta"] = eta_hx
        self.conns["yh"]["eta"] = eta_yh

        self.conns["hx"]["w"] = np.asmatrix(np.ones((dims[1], dims[0])))
        self.conns["yh"]["w"] = np.asmatrix(np.ones((dims[2], dims[1])))
        self.conns["ih"]["w"] = np.asmatrix(np.ones((dims[2], dims[1])))
        self.conns["hi"]["w"] = np.asmatrix(np.ones((dims[1], dims[2])))
        self.conns["hy"]["w"] = np.asmatrix(np.ones((dims[1], dims[2])))

        self.conns["hx"]["dt_w"] = np.asmatrix(np.zeros((dims[1], dims[0])))
        self.conns["yh"]["dt_w"] = np.asmatrix(np.zeros((dims[2], dims[1])))
        self.conns["ih"]["dt_w"] = np.asmatrix(np.zeros((dims[2], dims[1])))
        self.conns["hi"]["dt_w"] = np.asmatrix(np.zeros((dims[1], dims[2])))
        self.conns["hy"]["dt_w"] = np.asmatrix(np.zeros((dims[1], dims[2])))

        self.conns["hx"]["t_w"] = np.asmatrix(np.zeros((dims[1], dims[0])))
        self.conns["yh"]["t_w"] = np.asmatrix(np.zeros((dims[2], dims[1])))
        self.conns["ih"]["t_w"] = np.asmatrix(np.zeros((dims[2], dims[1])))
        self.conns["hi"]["t_w"] = np.asmatrix(np.zeros((dims[1], dims[2])))
        self.conns["hy"]["t_w"] = np.asmatrix(np.zeros((dims[1], dims[2])))

        self.hx_teacher = np.asmatrix(np.random.random((dims[1], dims[0])) * 2 - 1)
        # equivalent to dividing by gamma?
        self.yh_teacher = np.asmatrix(np.random.random((dims[2], dims[1])) * 2 - 1) * 10

    def simulate(self, T):
        for i in range(int(T/delta_t)):

            delta_u_x = -self.U_x + self.I_x
            delta_u_h = -(g_l + g_d + g_a) * self.U_h + g_d * self.V_bh + g_a * self.V_ah
            delta_u_y = -(g_l + g_d + g_a) * self.U_y + g_d * self.V_by + g_s * phi_inverse(self.y)
            delta_u_i = -(g_l + g_d + g_a) * self.U_i + g_d * self.V_bi + g_si * self.U_y

            self.conns["hx"]["dt_w"] = -self.conns["hx"]["t_w"] + \
                np.outer(self.r_h - phi((g_d * self.V_bh)/(g_l + g_d + g_a)), self.r_x)
            self.conns["yh"]["dt_w"] = -self.conns["yh"]["t_w"] + \
                np.outer(self.r_y - phi((g_d * self.V_by)/(g_l + g_d)), self.r_h)
            self.conns["hy"]["dt_w"] = -self.conns["hy"]["t_w"] + \
                np.outer(self.r_h - phi(self.conns["hy"]["w"] * self.r_y.T).T, self.r_y)
            self.conns["ih"]["dt_w"] = -self.conns["ih"]["t_w"] + \
                np.outer(self.r_i - phi((g_d * self.V_bi)/(g_l + g_d)), self.r_h)
            self.conns["hi"]["dt_w"] = -self.conns["hi"]["t_w"] + np.outer(-self.V_ah, self.r_i)

            self.U_x = self.U_x + (delta_t/tau_x) * delta_u_x
            # TODO: strictly speaking this should just be U_x but use_phi does not work as intended yet.
            self.r_x = phi(self.U_x)

            self.y = phi(self.yh_teacher * phi(self.hx_teacher * self.r_x.T)).T

            self.V_bh = (self.conns["hx"]["w"] * self.r_x.T).T
            self.V_ah = (self.conns["hy"]["w"] * self.r_y.T + self.conns["hi"]["w"] * self.r_i.T).T
            self.U_h = self.U_h + delta_t * delta_u_h
            self.r_h = phi(self.U_h)

            self.V_by = (self.conns["yh"]["w"] * self.r_h.T).T
            self.U_y = self.U_y + delta_t * delta_u_y
            self.r_y = phi(self.U_y)

            self.V_bi = (self.conns["ih"]["w"] * self.r_h.T).T
            self.U_i = self.U_i + delta_t * delta_u_i
            self.r_i = phi(self.U_i)

            for name, d in self.conns.items():
                d["t_w"] = d["t_w"] + (delta_t/tau_delta) * d["dt_w"]
                d["w"] = d["w"] + d["eta"] * delta_t * d["t_w"]
                d["record"].append(d["w"])

            self.V_ah_record.append(self.V_ah)
            self.U_i_record.append(self.U_i)
            self.U_y_record.append(self.U_y)

    def set_input(self, input_currents):
        self.I_x = input_currents

    def set_target(self, indices):
        for i in range(self.dims[-1]):
            if i in indices:
                self.pyr_pops[-1][i].set({"soma": {"I_e": self.stim_amp}})
            else:
                self.pyr_pops[-1][i].set({"soma": {"I_e": 0}})
