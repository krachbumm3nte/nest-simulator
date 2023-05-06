from copy import deepcopy

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from src.networks.layer import Layer, OutputLayer
from src.networks.network import Network
from src.params import Params


class NumpyNetwork(Network):

    def __init__(self, p: Params) -> None:
        super().__init__(p)
        self.output_loss = []
        self.u_target = np.zeros(self.dims[-1])
        self.r_in = np.zeros(self.dims[0])
        self.setup_populations()
        self.clear_records()

    def setup_populations(self):
        eta = {}
        for i in range(len(self.dims)-2):
            eta["up"] = self.p.eta["up"][i]
            eta["pi"] = self.p.eta["pi"][i]
            eta["ip"] = self.p.eta["ip"][i]
            self.layers.append(Layer(self.p, self, i))
        eta["up"] = self.p.eta["up"][-1]
        eta["pi"] = self.p.eta["pi"][-1]
        eta["ip"] = self.p.eta["ip"][-1]
        self.layers.append(OutputLayer(self.p, self, len(self.dims)-2))

        if self.p.init_self_pred:
            for i in range(len(self.layers) - 1):
                layer = self.layers[i]
                l_next = self.layers[i + 1]
                layer.W_pi = -layer.W_down.copy()
                layer.W_ip = l_next.W_up.copy() * l_next.gb / (l_next.gl + l_next.ga + l_next.gb) * \
                    (layer.gl + layer.gd) / layer.gd

    def clear_records(self):
        self.weight_record = self.copy_weights()
        for i, weight_dict in enumerate(self.weight_record):
            for key, weights in weight_dict.items():
                self.weight_record[i][key] = np.expand_dims(weights, axis=0)
        self.V_ah_record = np.zeros((1, self.dims[-2]))
        self.V_bh_record = np.zeros((1, self.dims[-2]))
        self.U_h_record = np.zeros((1, self.dims[-2]))
        self.U_i_record = np.zeros((1, self.dims[-1]))
        self.V_bi_record = np.zeros((1, self.dims[-1]))
        self.U_y_record = np.zeros((1, self.dims[-1]))
        self.V_by_record = np.zeros((1, self.dims[-1]))
        self.U_x_record = np.zeros((1, self.dims[0]))

    def copy_weights(self):
        weights = []
        for n in range(len(self.layers) - 1):
            layer = self.layers[n]
            weights.append({"up": layer.W_up.copy(),
                            "pi": layer.W_pi.copy(),
                            "ip": layer.W_ip.copy(),
                            "down": layer.W_down.copy()})
        weights.append({"up": self.layers[-1].W_up.copy()})
        return weights

    def train_batch(self, x_batch, y_batch):
        loss = []
        n_samples = int((self.p.t_pres - self.p.out_lag)/self.record_interval)

        for x, y in zip(x_batch, y_batch):
            self.reset()
            self.set_input(x)
            self.set_target(y)
            self.simulate(self.t_pres, enable_recording=True)
            y_pred = np.mean(self.U_y_record[-n_samples:], axis=0)
            loss.append(mse(y_pred, y))

        if self.p.store_errors:
            U_I = np.mean(self.U_i_record[-n_samples:], axis=0)
            V_ah = np.mean(self.V_ah_record[-n_samples:], axis=0)
            self.apical_error.append((self.epoch, float(np.linalg.norm(V_ah))))
            self.intn_error.append([self.epoch, mse(self.phi(U_I), self.phi(y_pred))])

        return np.mean(loss)

    def set_target(self, target_seq):
        self.target_seq = target_seq

    def test_batch(self, x_batch, y_batch):
        acc = []
        loss_mse = []
        for x_test, y_actual in zip(x_batch, y_batch):
            self.set_input(x_test)
            self.set_target(np.zeros(self.dims[-1]))
            self.simulate(self.t_pres, True, False)
            y_pred = np.mean(self.U_y_record[int((self.p.out_lag/self.t_pres)*self.record_interval):], axis=0)

            loss_mse.append(mse(y_actual, y_pred))
            acc.append(np.argmax(y_actual) == np.argmax(y_pred))
            self.reset()
            self.clear_records()

        return np.mean(acc), np.mean(loss_mse)

    def simulate(self, t, enable_recording=False, plasticity=True):
        for i in range(int(t/self.dt)):
            self.r_in += (self.dt/self.tau_x) * (self.I_x - self.r_in)
            self.u_target += (self.dt/self.tau_x) * (self.target_seq - self.u_target)

            if self.le:
                self.layers[0].update(self.r_in, self.layers[1].u_pyr["forw"], plasticity)
                for n in range(1, len(self.layers) - 1):
                    self.layers[n].update(self.phi(self.layers[n - 1].u_pyr["forw"]), self.layers[n + 1].u_pyr["forw"],
                                          plasticity)
                self.layers[-1].update(self.phi(self.layers[-2].u_pyr["forw"]),
                                       self.u_target, plasticity)
            else:
                self.layers[0].update(self.r_in, self.layers[1].u_pyr["soma"], plasticity)
                for n in range(1, len(self.layers) - 1):
                    self.layers[n].update(self.phi(self.layers[n - 1].u_pyr["soma"]), self.layers[n + 1].u_pyr["soma"],
                                          plasticity)
                self.layers[-1].update(self.phi(self.layers[-2].u_pyr["soma"]),
                                       self.u_target, plasticity)

            for layer in self.layers:
                layer.apply(plasticity)

            if enable_recording:
                if self.iteration*self.dt % self.record_interval < self.dt:
                    self.record_state()
            self.iteration += 1

    def record_state(self):
        U_y = self.layers[-1].u_pyr["soma"]
        self.U_y_record = np.concatenate((self.U_y_record, np.expand_dims(U_y, 0)), axis=0)
        if self.p.store_errors:
            self.V_by_record = np.concatenate(
                (self.V_by_record, np.expand_dims(self.layers[-1].u_pyr["basal"], 0)), axis=0)
            self.V_ah_record = np.concatenate(
                (self.V_ah_record, np.expand_dims(self.layers[-2].u_pyr["apical"], 0)), axis=0)
            self.V_bh_record = np.concatenate(
                (self.V_bh_record, np.expand_dims(self.layers[-2].u_pyr["basal"], 0)), axis=0)
            self.U_h_record = np.concatenate(
                (self.U_h_record, np.expand_dims(self.layers[-2].u_pyr["soma"], 0)), axis=0)
            self.U_i_record = np.concatenate(
                (self.U_i_record, np.expand_dims(self.layers[-2].u_inn["soma"], 0)), axis=0)
            self.V_bi_record = np.concatenate(
                (self.V_bi_record, np.expand_dims(self.layers[-2].u_inn["dendrite"], 0)), axis=0)
            self.U_x_record = np.concatenate((self.U_x_record, np.expand_dims(self.r_in, 0)), axis=0)

        if self.p.record_weights:
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
        for layer in self.layers[:-1]:
            weights.append({"up": layer.W_up.copy(),
                            "pi": layer.W_pi.copy(),
                            "ip": layer.W_ip.copy(),
                            "down": layer.W_down.copy()})
        weights.append({"up": self.layers[-1].W_up.copy()})
        return weights

    def reset(self):
        self.set_input(np.zeros(self.dims[0]))
        self.I_x = np.zeros(self.dims[0])

        self.set_target(np.zeros(self.dims[-1]))
        self.u_target = np.zeros(self.dims[-1])

        for layer in self.layers:
            layer.reset()
        self.clear_records()

    def set_all_weights(self, weights):
        print("setting all network weights... ", end="")
        for i, w in enumerate(weights[:-1]):
            self.layers[i].W_up = w["up"].copy()
            self.layers[i].W_down = w["down"].copy()
            self.layers[i].W_pi = w["pi"].copy()
            self.layers[i].W_ip = w["ip"].copy()

        self.layers[-1].W_up = weights[-1]["up"].copy()
        self.clear_records()
        print("Done")
