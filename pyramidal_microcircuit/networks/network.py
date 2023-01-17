from copy import deepcopy
import numpy as np

class Network:

    def __init__(self, sim, nrn, syns) -> None:
        self.sim = deepcopy(sim)  # simulation parameters
        self.nrn = deepcopy(nrn)  # neuron parameters
        self.syns = deepcopy(syns)  # synapse parameters

        self.dims = sim["dims"]
        self.iteration = 0
        self.sigma_noise = sim["sigma"]

        self.phi = nrn["phi"]
        self.phi_inverse = nrn["phi_inverse"]

        self.teacher = sim["teacher"]
        if self.teacher:
            self.hx_teacher = self.gen_weights(self.dims[0], self.dims[1], True)
            self.yh_teacher = self.gen_weights(self.dims[1], self.dims[2], True) / self.nrn["gamma"]
            self.y = np.random.random(self.dims[2])

    def gen_weights(self, lr, next_lr, matrix = False):
        weights = np.random.uniform(self.syns["wmin_init"], self.syns["wmax_init"], (next_lr, lr))
        return np.asmatrix(weights) if matrix else weights
