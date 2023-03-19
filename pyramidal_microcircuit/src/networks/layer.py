
from abc import ABC, abstractmethod

import numpy as np

dtype = np.float32


class AbstractLayer():
    def __init__(self, p, net, layer) -> None:
        self.layer = layer
        self.p = p
        self.net = net
        self.ga = p.g_a
        self.gb = p.g_d  # TODO: separate these?
        self.gd = p.g_d
        self.gl = p.g_l
        self.gsom = p.g_som
        self.tau_x = p.tau_x
        self.le = p.latent_equilibrium

        self.noise_factor = p.noise_factor if p.noise else 0
        self.tau_delta = p.tau_delta
        self.dt = p.delta_t

        self.leakage = self.gl + self.ga + self.gb
        self.lambda_out = p.lambda_out
        self.lambda_ah = p.lambda_ah
        self.lambda_bh = p.lambda_bh

        self.Wmin = -4
        self.Wmax = 4  # TODO: read from config
        self.gamma = p.gamma
        self.beta = p.beta
        self.theta = p.theta
        self.weight_scale = p.weight_scale if p.spiking else 1
        self.eta = {
            "up": p.eta["up"][self.layer],
            "ip": p.eta["ip"][self.layer],
            "pi": p.eta["pi"][self.layer],
            "down": p.eta["down"][self.layer],
        }

    @abstractmethod
    def update(self, r_in, u_next, plasticity, noise_on=False):
        pass

    @abstractmethod
    def apply(self, plasticity):
        pass

    @abstractmethod
    def reset(self, reset_weights=False):
        pass

    def gen_weights(self, n_in, n_out, wmin=None, wmax=None):
        if not wmin:
            wmin = -1/self.weight_scale
        if not wmax:
            wmax = 1/self.weight_scale
        return np.random.uniform(wmin, wmax, (n_out, n_in))

    def phi(self, x, thresh=15):

        res = x.copy()
        ind = np.abs(x) < thresh
        res[x < -thresh] = 0
        res[ind] = self.gamma * np.log(1 + np.exp(self.beta * (x[ind] - self.theta)))
        return res


class Layer(AbstractLayer):

    def __init__(self, p, net, layer) -> None:
        super().__init__(p, net, layer)

        self.N_in = net.dims[layer]
        self.N_pyr = net.dims[layer+1]
        self.N_next = net.dims[layer+2]

        self.W_down = self.gen_weights(self.N_next, self.N_pyr, -1, 1)
        self.W_up = self.gen_weights(self.N_in, self.N_pyr, -1, 1)
        self.W_pi = self.gen_weights(self.N_next, self.N_pyr, -1, 1)
        self.W_ip = self.gen_weights(self.N_pyr, self.N_next, -1, 1)

        self.u_pyr = {"basal": np.zeros(self.N_pyr, dtype=dtype), "apical": np.zeros(self.N_pyr, dtype=dtype),
                      "soma": np.zeros(self.N_pyr, dtype=dtype), "forw": np.ones(self.N_pyr, dtype=dtype),
                      "steadystate": np.zeros(self.N_pyr, dtype=dtype), "udot": np.zeros(self.N_pyr, dtype=dtype)}
        self.u_inn = {"dendrite": np.zeros(self.N_next, dtype=dtype), "soma": np.zeros(self.N_next, dtype=dtype),
                      "forw": np.zeros(self.N_next, dtype=dtype), "steadystate": np.zeros(self.N_next, dtype=dtype),
                      "udot": np.zeros(self.N_next, dtype=dtype)}

        self.Delta_up = np.zeros((self.N_pyr, self.N_in), dtype=dtype)
        self.Delta_pi = np.zeros((self.N_pyr, self.N_next), dtype=dtype)
        self.Delta_ip = np.zeros((self.N_next, self.N_pyr), dtype=dtype)

    def update(self, r_in, u_next, plasticity, noise_on=False):
        r_next = self.phi(u_next)
        r_pyr = self.phi(self.u_pyr["forw"]) if self.le else self.phi(self.u_pyr["soma"])
        r_inn = self.phi(self.u_inn["forw"]) if self.le else self.phi(self.u_inn["soma"])
        self.u_pyr["basal"][:] = self.W_up @ r_in  # [:] to enforce rhs to be copied to lhs
        self.u_pyr["apical"][:] = self.W_down @ r_next + self.W_pi @ r_inn
        self.u_inn["dendrite"][:] = self.W_ip @ r_pyr

        self.u_pyr["steadystate"][:] = (self.gb * self.u_pyr["basal"] + self.ga * self.u_pyr["apical"]) / (
            self.gl + self.gb + self.ga)
        # inter has no apical, if more than one hidden layer: make sure that g_som = g_apial(above pyr)
        self.u_inn["steadystate"][:] = (self.gb * self.u_inn["dendrite"] + self.gsom * u_next) / (
            self.gl + self.gb + self.gsom)

        u_p = self.u_pyr["soma"]
        u_i = self.u_inn["soma"]

        self.u_pyr["udot"][:] = (self.gl + self.gb + self.ga) * (self.u_pyr["steadystate"] - u_p)
        self.du_pyr = self.u_pyr["udot"] * self.dt
        self.u_inn["udot"][:] = (self.gl + self.gb + self.gsom) * (self.u_inn["steadystate"] - u_i)
        self.du_inn = self.u_inn["udot"] * self.dt

        if plasticity:
            gtot = self.gl + self.gb + self.ga

            if self.le:
                u_new_pyr = self.u_pyr["soma"] + self.u_pyr["udot"] / gtot
                u_new_inn = self.u_inn["soma"] + self.u_inn["udot"] / (self.gl + self.gb + self.gsom)
            else:
                u_new_pyr = self.u_pyr["soma"] + self.du_pyr
                u_new_inn = self.u_inn["soma"] + self.du_inn

            self.Delta_up = np.outer(
                self.phi(u_new_pyr) - self.phi(self.gb / gtot * self.u_pyr["basal"]), r_in)
            self.Delta_ip = np.outer(
                self.phi(u_new_inn) - self.phi(self.gb / (self.gl + self.gb) * self.u_inn["dendrite"]), r_pyr)
            self.Delta_pi = np.outer(-self.u_pyr["apical"], r_inn)

    def apply(self, plasticity):
        if self.le:
            tau_pyr = 1. / (self.gl + self.gb + self.ga)
            tau_inn = 1. / (self.gl + self.gb + self.gsom)
            # important: update u_forw before updating u_soma!
            self.u_pyr["forw"][:] = self.u_pyr["soma"] + tau_pyr * self.u_pyr["udot"]
            self.u_inn["forw"][:] = self.u_inn["soma"] + tau_inn * self.u_inn["udot"]
        self.u_pyr["soma"] += self.du_pyr
        self.u_inn["soma"] += self.du_inn
        # apply weight updates
        if plasticity:
            self.W_up += self.dt * self.eta["up"] * self.Delta_up
            self.W_up = np.clip(self.W_up, self.Wmin, self.Wmax)
            self.W_ip += self.dt * self.eta["ip"] * self.Delta_ip
            self.W_pi += self.dt * self.eta["pi"] * self.Delta_pi

    def reset(self, reset_weights=False):
        '''
        Reset all potentials and Deltas (weight update matrices) to zero.
        Parameters
        ----------
        reset_weights:  Also draw weights again from random distribution.

        '''
        self.u_pyr["basal"].fill(0)
        self.u_pyr["soma"].fill(0)
        self.u_pyr["apical"].fill(0)
        self.u_pyr["steadystate"].fill(0)
        self.u_pyr["forw"].fill(0)
        self.u_pyr["udot"].fill(0)
        self.u_inn["dendrite"].fill(0)
        self.u_inn["soma"].fill(0)
        self.u_inn["steadystate"].fill(0)
        self.u_inn["forw"].fill(0)
        self.u_inn["udot"].fill(0)

        if reset_weights:
            self.W_up = self.gen_weights(self.N_in, self.N_pyr, -1, 1)
            self.W_down = self.gen_weights(self.N_next, self.N_pyr, -1, 1)
            self.W_pi = self.gen_weights(self.N_next, self.N_pyr, -1, 1)
            self.W_ip = self.gen_weights(self.N_pyr, self.N_next, -1, 1)

        self.Delta_up.fill(0)
        self.Delta_pi.fill(0)
        self.Delta_ip.fill(0)


class OutputLayer(AbstractLayer):

    def __init__(self, net, p, layer) -> None:
        super().__init__(net, p, layer)
        self.ga = 0
        self.N_in = net.dims[-2]
        self.N_out = net.dims[-1]
        self.u_pyr = {"basal": np.zeros(self.N_out, dtype=dtype), "soma": np.zeros(self.N_out, dtype=dtype),
                      "steadystate": np.zeros(self.N_out, dtype=dtype), "forw": np.zeros(self.N_out, dtype=dtype),
                      "udot": np.zeros(self.N_out, dtype=dtype)}

        self.W_up = self.gen_weights(self.N_in, self.N_out, -1, 1)
        self.Delta_up = np.zeros((self.N_out, self.N_in), dtype=dtype)

    def update(self, r_in, u_next, plasticity, noise_on=False):
        self.u_pyr["basal"][:] = self.W_up @ r_in  # [:] to enforce rhs to be copied to lhs

        if u_next.any() > 0:
            self.u_pyr["steadystate"] = (self.gb * self.u_pyr["basal"] + self.gsom * u_next) / (
                self.gl + self.gb + self.gsom)
        else:
            self.u_pyr["steadystate"] = (self.gb * self.u_pyr["basal"]) / (
                self.gl + self.gb)

        # compute changes

        # if self.le:
        #    self.u_pyr["udot"] = (self.gl + self.gb + self.gsom) * (self.u_pyr["steadystate"] - self.u_pyr["soma"])
        #    self.du_pyr = self.u_pyr["udot"] * self.dt
        # else:
        self.u_pyr["udot"] = (self.gl + self.gb + self.gsom) * (self.u_pyr["steadystate"] - self.u_pyr["soma"])
        self.du_pyr = self.u_pyr["udot"] * self.dt

        if plasticity:
            gtot = self.gl + self.gb
            if self.le:
                u_new_pyr = self.u_pyr["soma"] + self.u_pyr["udot"] / (self.gl + self.gb + self.gsom)
            else:
                u_new_pyr = self.u_pyr["soma"] + self.du_pyr

            self.Delta_up = np.outer(
                self.phi(u_new_pyr) - self.phi(self.gb / gtot * self.u_pyr["basal"]), r_in)

    def apply(self, plasticity):
        if self.le:
            tau_pyr = 1. / (self.gl + self.gb + self.gsom)
            # important: update u_forw before updating u_soma!
            self.u_pyr["forw"][:] = self.u_pyr["soma"] + tau_pyr * self.u_pyr["udot"]
        self.u_pyr["soma"] += self.du_pyr
        # apply weight updates
        if plasticity:
            self.W_up += self.dt * self.eta["up"] * self.Delta_up
            self.W_up = np.clip(self.W_up, self.Wmin, self.Wmax)

    def reset(self, reset_weights=False):
        '''
        Reset all potentials and Deltas (weight update matrices) to zero.
        Parameters
        ----------
        reset_weights:  Also draw weights again from random distribution.

        '''
        self.u_pyr["basal"].fill(0)
        self.u_pyr["soma"].fill(0)
        self.u_pyr["steadystate"].fill(0)
        self.u_pyr["forw"].fill(0)
        self.u_pyr["udot"].fill(0)

        if reset_weights:
            self.W_up = self.gen_weights(self.N_in, self.N_out, -1, 1)

        self.Delta_up.fill(0)
