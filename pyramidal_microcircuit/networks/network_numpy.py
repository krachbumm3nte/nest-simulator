import numpy as np
from sklearn.metrics import mean_squared_error as mse
from time import time
from .network import Network
from copy import deepcopy
from .layer import Layer
from .outputlayer import OutputLayer


class NumpyNetwork(Network):

    def __init__(self, sim, nrn, syn) -> None:
        super().__init__(sim, nrn, syn)

        self.conns = {}

        self.V_ah_record = np.zeros((1, self.dims[-2]))
        self.V_bh_record = np.zeros((1, self.dims[-2]))
        self.U_h_record = np.zeros((1, self.dims[-2]))
        self.U_i_record = np.zeros((1, self.dims[-1]))
        self.U_y_record = np.zeros((1, self.dims[-1]))
        self.U_x_record = np.zeros((1, self.dims[0]))
        self.u_target = np.zeros(self.dims[-1])
        self.output_loss = []
        self.r_in = np.zeros(self.dims[0])
        self.setup_populations()

        self.setup_records()
        self.iteration = 0

    def setup_populations(self):
        eta = {}
        # construct all layers

        for i in range(len(self.dims) - 2):
            eta["up"] = self.syn["eta"]["up"][i]
            eta["pi"] = self.syn["eta"]["pi"][i]
            eta["ip"] = self.syn["eta"]["ip"][i]
            self.layers.append(Layer(self.nrn, self.sim, self.syn, i, eta))
        eta["up"] = self.syn["eta"]["up"][-1]
        eta["pi"] = self.syn["eta"]["pi"][-1]
        eta["ip"] = self.syn["eta"]["ip"][-1]
        self.layers.append(OutputLayer(self.nrn, self.sim, self.syn, eta))

        if self.sim["init_self_pred"]:
            for i in range(len(self.layers) - 1):
                l = self.layers[i]
                l_next = self.layers[i + 1]
                l.W_pi = -l.W_down.copy()
                l.W_ip = l_next.W_up.copy() * l_next.gb / (l_next.gl + l_next.ga + l_next.gb) * (l.gl + l.gd) / l.gd

    def setup_records(self):
        self.weight_record = self.copy_weights()
        for i, weight_dict in enumerate(self.weight_record):
            for key, weights in weight_dict.items():
                self.weight_record[i][key] = np.expand_dims(weights, axis=0)

    def train_teacher(self, T):
        input_currents = np.random.random(self.dims[0])

        self.set_input(input_currents)

        for i in range(int(T/self.dt)):
            self.simulate(self.target_teacher)

    def target_teacher(self):
        return self.get_teacher_output(self.r_in)

    def copy_weights(self):
        weights = []
        for n in range(len(self.layers) - 1):
            l = self.layers[n]
            weights.append({"up": l.W_up.copy(),
                            "pi": l.W_pi.copy(),
                            "ip": l.W_ip.copy(),
                            "down": l.W_down.copy()})
        weights.append({"up": self.layers[-1].W_up.copy()})
        return weights

    def test_teacher(self, T):
        for i in range(int(T/self.dt)):
            # do not inject output layer current during testing
            self.simulate(np.zeros(self.dims[-1]), False, False)
            self.output_pred = self.lambda_out * self.conns["yh"]["w"] @ self.phi(self.lambda_bh * self.V_bh)
            # TODO: fix scaling between teacher and predicted output!
            self.output_loss.append(mse(np.asarray(self.u_target), self.output_pred))

    def train_epoch(self, x_batch, y_batch):
        for x_train, y_train in zip(x_batch, y_batch):
            self.set_input(x_train)
            self.target_seq = y_train
            for i in range(int(self.sim_time/self.dt)):
                self.simulate(self.target_filtered)
        self.reset()

    def test_bars(self, n_samples=8):
        acc = []
        loss_mse = []
        for sample_idx in range(n_samples):
            x_test, y_actual = self.generate_bar_data()
            self.set_input(x_test)
            for i in range(int(self.sim_time/self.dt)):
                self.simulate(lambda: np.zeros(self.dims[-1]), False, False)
            y_pred = self.layers[-1].u_pyr["soma"]
            loss_mse.append(mse(y_actual, y_pred))
            acc.append(np.argmax(y_actual) == np.argmax(y_pred))
            self.reset()

        self.test_acc.append(np.mean(acc))
        self.test_loss.append(np.mean(loss_mse))

    def target_filtered(self):
        u_old = deepcopy(self.u_target)
        self.u_target += self.dt/self.tau_x * (self.target_seq - self.u_target)
        return u_old if self.le else self.u_target

    def simulate(self, train_function, enable_recording=True, plasticity=True):
        noise_on = False
        self.r_in = self.r_in + (self.dt/self.tau_x) * (self.I_x - self.r_in)
        if self.le:
            self.layers[0].update(self.r_in, self.layers[1].u_pyr["forw"], plasticity, noise_on=noise_on)
            for n in range(1, len(self.layers) - 1):
                self.layers[n].update(self.phi(self.layers[n - 1].u_pyr["forw"]), self.layers[n + 1].u_pyr["forw"],
                                      plasticity, noise_on=noise_on)
            self.layers[-1].update(self.phi(self.layers[-2].u_pyr["forw"]),
                                   train_function(), plasticity, noise_on=noise_on)
        else:
            self.layers[0].update(self.r_in, self.layers[1].u_pyr["soma"], plasticity, noise_on=noise_on)
            for n in range(1, len(self.layers) - 1):
                self.layers[n].update(self.phi(self.layers[n - 1].u_pyr["soma"]), self.layers[n + 1].u_pyr["soma"],
                                      plasticity, noise_on=noise_on)
            self.layers[-1].update(self.phi(self.layers[-2].u_pyr["soma"]),
                                   train_function(), plasticity, noise_on=noise_on)

        for layer in self.layers:
            layer.apply(plasticity)

        if enable_recording:
            if self.iteration*self.dt % self.record_interval < self.dt:
                self.record_state()
        self.iteration += 1

    def record_state(self):
        U_y = self.layers[-1].u_pyr["soma"]
        self.U_y_record = np.concatenate((self.U_y_record, np.expand_dims(U_y, 0)), axis=0)
        self.V_ah_record = np.concatenate((self.V_ah_record, np.expand_dims(self.layers[-2].u_pyr["apical"], 0)), axis=0)
        self.V_bh_record = np.concatenate((self.V_bh_record, np.expand_dims(self.layers[-2].u_pyr["basal"], 0)), axis=0)
        self.U_i_record = np.concatenate((self.U_i_record, np.expand_dims(self.layers[-2].u_inn["soma"], 0)), axis=0)
        self.U_h_record = np.concatenate((self.U_h_record, np.expand_dims(self.layers[-2].u_pyr["soma"], 0)), axis=0)
        self.U_x_record = np.concatenate((self.U_x_record, np.expand_dims(self.r_in, 0)), axis=0)
        self.train_loss.append(mse(self.u_target, U_y))

        for i, weight_dict in enumerate(self.copy_weights()):
            for key, weights in weight_dict.items():
                self.weight_record[i][key] = np.concatenate(
                    (self.weight_record[i][key], np.expand_dims(weights, axis=0)), axis=0)

    def set_input(self, input_currents):
        """Inject a constant current into all neurons in the input layer.

        Arguments:
            input_currents -- Iterable of length equal to the input dimension.
        """
        self.I_x = input_currents

    def get_weight_dict(self):
        weights = []
        for l in self.layers[:-1]:
            weights.append({
                "up": l.W_up.copy(),
                "down": l.W_down.copy(),
                "pi": l.W_pi.copy(),
                "ip": l.W_ip.copy()
            })

        weights.append(
            {"up": self.layers[-1].W_up.copy()}
        )
        return weights

    def reset(self):
        for l in self.layers:
            l.reset()

    def set_weights(self, weights):
        for i, w in enumerate(weights[:-1]):
            self.layers[i].W_up = w["up"].copy()
            self.layers[i].W_down = w["down"].copy()
            self.layers[i].W_pi = w["pi"].copy()
            self.layers[i].W_ip = w["ip"].copy()

        self.layers[-1].W_up = weights[-1]["up"].copy()
        self.setup_records()