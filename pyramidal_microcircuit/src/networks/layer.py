# -*- coding: utf-8 -*-
#
# layer.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.


from abc import ABC, abstractmethod

import numpy as np

dtype = np.float32


class AbstractLayer(ABC):
    def __init__(self, p, net, layer) -> None:
        """Abstract base class for a Layer in the dendritic error network

        Arguments:
            p -- instance of params.Params
            net -- instance of networks.Network to which this layer belongs
            layer -- index of this layer (starting at 0 for the first hidden layer)
        """

        self.layer = layer
        self.p = p
        self.net = net
        self.ga = p.g_a
        self.gb = p.g_d
        self.gd = p.g_d
        self.gl = p.g_l
        self.gsom = p.g_som
        self.tau_x = p.tau_x
        self.le = p.latent_equilibrium

        self.tau_delta = p.tau_delta
        self.dt = p.delta_t

        self.leakage = self.gl + self.ga + self.gb

        self.Wmin = -4
        self.Wmax = 4
        self.gamma = p.gamma
        self.beta = p.beta
        self.theta = p.theta
        self.psi = p.psi if p.spiking else 1
        self.eta = {
            "up": p.eta["up"][self.layer],
            "ip": p.eta["ip"][self.layer],
            "pi": p.eta["pi"][self.layer],
            "down": p.eta["down"][self.layer],
        }

    def update(self, r_in, u_next, plasticity, noise_on=False):
        """Update the state of this layer from feedforward- and feedback inputs

        Arguments:
            r_in -- rate of previous layer pyramidal neurons
            u_next -- somatic voltage of next layer pyramidal neurons
            plasticity -- if true, compute weight changes

        Keyword Arguments:
            noise_on -- if true, inject noise into membranes (default: {False})
        """
        pass

    def apply(self, plasticity):
        """Apply changes computed in update() function

        Arguments:
            plasticity -- if true, update synaptic weights
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset membrane potentials and synaptic weight deltas
        """
        pass

    def gen_weights(self, n_in, n_out, wmin=None, wmax=None):
        """Generate a set of weights to initialize a synaptic population

        Arguments:
            n_in -- number of neuron in the source population
            n_out -- number of neurons in the target population

        Keyword Arguments:
            wmin -- minimum initial weight (default: {None})
            wmax -- maximum initial weight (default: {None})

        Returns:
            np.array of dimensions (n_out, n_in)
        """
        if wmin is None:
            wmin = self.p.wmin_init/self.psi
        if wmax is None:
            wmax = self.p.wmax_init/self.psi
        return np.random.uniform(wmin, wmax, (n_out, n_in))

    def phi(self, x, thresh=15):
        """Neuronal transfer function

        Arguments:
            x -- np.array of somatic voltages

        Keyword Arguments:
            thresh -- threshold beyond which the activation scales linearly (default: {15})

        Returns:
            np.array of equal dimensions as the input
        """

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
        # self.Delta_down = np.zeros((self.N_pyr, self.N_next), dtype=dtype)

        # pyramidal and interneuron effective time constants, used for prospective activation
        self.tau_pyr = self.p.C_m_som / (self.gl + self.gb + self.ga)
        self.tau_inn = self.p.C_m_som / (self.gl + self.gb + self.gsom)

    def update(self, r_in, u_next, plasticity):
        r_next = self.phi(u_next)
        r_pyr = self.phi(self.u_pyr["forw"]) if self.le else self.phi(self.u_pyr["soma"])
        r_inn = self.phi(self.u_inn["forw"]) if self.le else self.phi(self.u_inn["soma"])

        self.u_pyr["basal"][:] = self.W_up @ r_in
        v_api_dist = self.W_down @ r_next
        self.u_pyr["apical"][:] = v_api_dist + self.W_pi @ r_inn
        self.u_inn["dendrite"][:] = self.W_ip @ r_pyr

        self.u_pyr["steadystate"][:] = (self.gb * self.u_pyr["basal"] + self.ga * self.u_pyr["apical"]) * self.tau_pyr
        # inter has no apical, if more than one hidden layer: make sure that g_som = g_apial(above pyr)
        self.u_inn["steadystate"][:] = (self.gb * self.u_inn["dendrite"] + self.gsom * u_next) * self.tau_inn

        u_p = self.u_pyr["soma"]
        u_i = self.u_inn["soma"]

        self.u_pyr["udot"][:] = (self.u_pyr["steadystate"] - u_p) / self.tau_pyr
        self.du_pyr = self.u_pyr["udot"] * self.dt
        self.u_inn["udot"][:] = (self.u_inn["steadystate"] - u_i) / self.tau_inn
        self.du_inn = self.u_inn["udot"] * self.dt

        if plasticity:

            if self.le:
                u_new_pyr = self.u_pyr["soma"] + self.u_pyr["udot"] * self.tau_pyr
                u_new_inn = self.u_inn["soma"] + self.u_inn["udot"] * self.tau_inn
            else:
                u_new_pyr = self.u_pyr["soma"] + self.du_pyr
                u_new_inn = self.u_inn["soma"] + self.du_inn

            self.Delta_up = np.outer(
                self.phi(u_new_pyr) - self.phi(self.gb * self.tau_pyr * self.u_pyr["basal"]), r_in)
            self.Delta_ip = np.outer(
                self.phi(u_new_inn) - self.phi(self.gb / (self.gl + self.gb) * self.u_inn["dendrite"]), r_pyr)
            self.Delta_pi = np.outer(-self.u_pyr["apical"], r_inn)
            # self.Delta_down = np.outer(r_pyr - v_api_dist, r_next)

    def apply(self, plasticity):
        if self.le:
            # important: update u_forw before updating u_soma!
            self.u_pyr["forw"][:] = self.u_pyr["soma"] + self.tau_pyr * self.u_pyr["udot"]
            self.u_inn["forw"][:] = self.u_inn["soma"] + self.tau_inn * self.u_inn["udot"]
        self.u_pyr["soma"] += self.du_pyr
        self.u_inn["soma"] += self.du_inn
        # apply weight updates
        if plasticity:
            self.W_up += self.dt * self.eta["up"] * self.Delta_up
            self.W_ip += self.dt * self.eta["ip"] * self.Delta_ip
            self.W_pi += self.dt * self.eta["pi"] * self.Delta_pi
            # self.W_down += self.dt * self.eta["down"] * self.Delta_down
            self.W_up = np.clip(self.W_up, self.Wmin, self.Wmax)
            self.W_ip = np.clip(self.W_ip, self.Wmin, self.Wmax)
            self.W_pi = np.clip(self.W_pi, self.Wmin, self.Wmax)
            # self.W_down = np.clip(self.W_down, self.Wmin, self.Wmax)

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
        # self.Delta_down.fill(0)


class OutputLayer(AbstractLayer):

    def __init__(self, net, p, layer) -> None:
        """Output layer, which only contains pyramidal neurons

        Arguments:
            p -- instance of params.Params
            net -- instance of networks.Network to which this layer belongs
            layer -- index of this layer (starting at 0 for the first hidden layer)
        """
        super().__init__(net, p, layer)
        self.ga = 0
        self.N_in = net.dims[-2]
        self.N_out = net.dims[-1]
        self.u_pyr = {"basal": np.zeros(self.N_out, dtype=dtype), "soma": np.zeros(self.N_out, dtype=dtype),
                      "steadystate": np.zeros(self.N_out, dtype=dtype), "forw": np.zeros(self.N_out, dtype=dtype),
                      "udot": np.zeros(self.N_out, dtype=dtype)}

        self.W_up = self.gen_weights(self.N_in, self.N_out, -1, 1)
        self.Delta_up = np.zeros((self.N_out, self.N_in), dtype=dtype)

        self.tau_pyr = self.p.C_m_som / (self.gl + self.gb + self.gsom)

    def update(self, r_in, u_tgt, plasticity, noise_on=False):
        self.u_pyr["basal"][:] = self.W_up @ r_in  # [:] to enforce rhs to be copied to lhs

        if u_tgt.any() > 0:
            self.u_pyr["steadystate"] = (self.gb * self.u_pyr["basal"] + self.gsom * u_tgt) * self.tau_pyr
        else:
            self.u_pyr["steadystate"] = (self.gb * self.u_pyr["basal"]) / (self.gl + self.gb)

        # compute changes

        # if self.le:
        #    self.u_pyr["udot"] = (self.gl + self.gb + self.gsom) * (self.u_pyr["steadystate"] - self.u_pyr["soma"])
        #    self.du_pyr = self.u_pyr["udot"] * self.dt
        # else:
        self.u_pyr["udot"] = (self.u_pyr["steadystate"] - self.u_pyr["soma"]) / self.tau_pyr
        self.du_pyr = self.u_pyr["udot"] * self.dt

        if plasticity:
            gtot = self.gl + self.gb
            if self.le:
                u_new_pyr = self.u_pyr["soma"] + self.u_pyr["udot"] * self.tau_pyr
            else:
                u_new_pyr = self.u_pyr["soma"] + self.du_pyr

            self.Delta_up = np.outer(
                self.phi(u_new_pyr) - self.phi(self.gb / gtot * self.u_pyr["basal"]), r_in)

    def apply(self, plasticity):
        if self.le:
            # important: update u_forw before updating u_soma!
            self.u_pyr["forw"][:] = self.u_pyr["soma"] + self.tau_pyr * self.u_pyr["udot"]
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
