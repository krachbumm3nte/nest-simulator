
import numpy as np
from .layer import AbstractLayer
dtype = np.float32


class OutputLayer(AbstractLayer):

    def __init__(self, nrn, sim, syn, eta) -> None:
        super().__init__(nrn, sim, syn, eta)
        self.ga = 0
        self.N_in = sim["dims"][-2]
        self.N_out = sim["dims"][-1]
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
