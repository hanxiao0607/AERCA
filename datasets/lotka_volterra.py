import numpy as np
from tqdm import tqdm
import os
from numba import njit


class LotkaVolterra:
    def __init__(self, options):
        """
        Dynamical multi-species Lotka–Volterra system.
        The original two-species Lotka–Volterra is a special case with p = 1 and d = 1.

        @param options: Dictionary with keys:
            - num_vars: Total number of variables (even number: prey and predator).
            - d: Number of GC parents per variable.
            - dt: Time step.
            - training_size: Number of training simulations.
            - testing_size: Number of testing simulations.
            - T: Total number of time steps.
            - seed: Random seed.
            - downsample_factor: Factor for downsampling.
            - data_dir: Directory for saving data.
            - mul: Multiplier for perturbation.
            - alpha_lv: Self-interaction strength of a prey species.
            - beta_lv: Predator-to-prey interaction strength.
            - gamma_lv: Self-interaction strength of a predator species.
            - delta_lv: Prey-to-predator interaction strength.
            - sigma_lv: Scale parameter for the noise.
            - adlength: Length of the adversarial perturbation.
            - adtype: Type of adversarial perturbation ('non_causal' or other).
        """
        self.options = options
        self.data_dict = {}
        self.p = options['num_vars'] // 2
        self.d = options['d']
        self.dt = options['dt']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.seed = options['seed']
        self.downsample_factor = options['downsample_factor']
        self.data_dir = options['data_dir']
        self.mul = options['mul']

        # Coupling strengths
        self.alpha = options['alpha_lv']
        self.beta = options['beta_lv']
        self.gamma = options['gamma_lv']
        self.delta = options['delta_lv']
        self.sigma = options['sigma_lv']
        self.adlength = options['adlength']
        self.adtype = options['adtype']

    def generate_example(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        lst_n = []
        lst_ab = []
        eps_n = []
        eps_ab = []
        lst_labels = []

        # Loop over each simulation
        for _ in tqdm(range(self.n)):
            # Generate simulation-specific adversarial timing and feature selection
            base_t_p = np.random.randint(int(0.5 * self.t / self.downsample_factor),
                                         self.t // self.downsample_factor, size=1)
            if self.adlength > 1:
                t_p = base_t_p + np.arange(self.adlength)
            else:
                t_p = base_t_p
            # Convert to original time indices (adjusted by downsampling) and subtract one
            trigger_indices = set(((t_p * self.downsample_factor) - 1).flatten())

            pp_p = np.random.randint(0, 2, size=1)
            n_features = np.random.randint(3, min(5, self.p) + 1) if self.p >= 3 else np.random.randint(2, min(5,
                                                                                                               self.p) + 1)
            feature_p = np.random.permutation(self.p)[:n_features]

            # Initial conditions
            xs_0 = np.random.uniform(10, 20, size=self.p)
            ys_0 = np.random.uniform(10, 20, size=self.p)

            xs = np.empty((self.t, self.p))
            ys = np.empty((self.t, self.p))
            eps_x = np.empty((self.t, self.p))
            eps_y = np.empty((self.t, self.p))
            xs[0] = xs_0
            ys[0] = ys_0

            xs_ab = np.empty((self.t, self.p))
            ys_ab = np.empty((self.t, self.p))
            eps_x_ab = np.empty((self.t, self.p))
            eps_y_ab = np.empty((self.t, self.p))
            label_x = np.zeros((self.t, self.p))
            label_y = np.zeros((self.t, self.p))
            xs_ab[0] = xs_0
            ys_ab[0] = ys_0

            count = 0
            for k in range(self.t - 1):
                if k in trigger_indices:
                    (xs[k + 1], ys[k + 1],
                     eps_x[k + 1], eps_y[k + 1],
                     xs_ab[k + 1], ys_ab[k + 1],
                     eps_x_ab[k + 1], eps_y_ab[k + 1],
                     label_x[k + 1], label_y[k + 1]) = self.next(
                        xs[k], ys[k], xs_ab[k], ys_ab[k],
                        self.dt, ab=1, pp_p=pp_p,
                        feature_p=feature_p, adtype=self.adtype, seq_k=count)
                    count += 1
                else:
                    (xs[k + 1], ys[k + 1],
                     eps_x[k + 1], eps_y[k + 1],
                     xs_ab[k + 1], ys_ab[k + 1],
                     eps_x_ab[k + 1], eps_y_ab[k + 1],
                     label_x[k + 1], label_y[k + 1]) = self.next(
                        xs[k], ys[k], xs_ab[k], ys_ab[k], self.dt)

            # Downsample and collect results
            ds_slice = slice(None, None, self.downsample_factor)
            lst_n.append(np.concatenate((xs[ds_slice], ys[ds_slice]), axis=1))
            eps_n.append(np.concatenate((eps_x[ds_slice], eps_y[ds_slice]), axis=1))
            lst_ab.append(np.concatenate((xs_ab[ds_slice], ys_ab[ds_slice]), axis=1))
            eps_ab.append(np.concatenate((eps_x_ab[ds_slice], eps_y_ab[ds_slice]), axis=1))
            lst_labels.append(np.concatenate((label_x[ds_slice], label_y[ds_slice]), axis=1))

        # Construct causal structure matrices using integer arithmetic instead of np.floor
        causal_struct = np.zeros((self.p * 2, self.p * 2))
        signed_causal_struct = np.zeros((self.p * 2, self.p * 2))
        for j in range(self.p):
            # Self-causation
            causal_struct[j, j] = 1
            causal_struct[j + self.p, j + self.p] = 1
            signed_causal_struct[j, j] = 1
            signed_causal_struct[j + self.p, j + self.p] = -1

            base_index = ((j + self.d) // self.d) * self.d
            # Prey-to-predator influence (adjust indices accordingly)
            causal_struct[j, base_index + self.p - self.d: base_index + self.p] = 1
            signed_causal_struct[j, base_index + self.p - self.d: base_index + self.p] = -1
            # Predator influenced by prey
            causal_struct[j + self.p, base_index - self.d: base_index] = 1
            signed_causal_struct[j + self.p, base_index - self.d: base_index] = 1

        self.data_dict = {
            'x_n_list': np.array(lst_n)[:, 50:, :],
            'eps_n_list': np.array(eps_n)[:, 50:, :],
            'x_ab_list': np.array(lst_ab)[:, 50:, :],
            'eps_ab_list': np.array(eps_ab)[:, 50:, :],
            'label_list': np.array(lst_labels)[:, 50:],
            'causal_struct': causal_struct,
            'signed_causal_struct': signed_causal_struct
        }

    def next(self, x, y, x_ab, y_ab, dt, ab=0, pp_p=0, feature_p=None, adtype='non_causal', seq_k=0):
        dt2 = dt / 2.0
        dt6 = dt / 6.0

        if ab == 1:
            label_x = np.zeros(self.p)
            label_y = np.zeros(self.p)
            xdot1, ydot1 = LotkaVolterra.f(x, y, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot2, ydot2 = LotkaVolterra.f(x + xdot1 * dt2, y + ydot1 * dt2, self.alpha, self.beta, self.gamma,
                                                self.delta, self.p, self.d)
            xdot3, ydot3 = LotkaVolterra.f(x + xdot2 * dt2, y + ydot2 * dt2, self.alpha, self.beta, self.gamma,
                                                self.delta, self.p, self.d)
            xdot4, ydot4 = LotkaVolterra.f(x + xdot3 * dt, y + ydot3 * dt, self.alpha, self.beta, self.gamma,
                                                self.delta, self.p, self.d)

            # Add noise to simulations
            eps_x = np.random.normal(scale=self.sigma, size=self.p)
            eps_y = np.random.normal(scale=self.sigma, size=self.p)
            eps_x_ab = eps_x.copy()
            eps_y_ab = eps_y.copy()
            xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt6 + eps_x
            ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt6 + eps_y

            if adtype == 'non_causal':
                xdot1_ab, ydot1_ab = LotkaVolterra.f(x_ab, y_ab, self.alpha, self.beta, self.gamma, self.delta,
                                                          self.p, self.d)
                xdot2_ab, ydot2_ab = LotkaVolterra.f(x_ab + xdot1_ab * dt2, y + ydot1_ab * dt2, self.alpha,
                                                          self.beta, self.gamma, self.delta, self.p, self.d)
                xdot3_ab, ydot3_ab = LotkaVolterra.f(x_ab + xdot2_ab * dt2, y_ab + ydot2_ab * dt2, self.alpha,
                                                          self.beta, self.gamma, self.delta, self.p, self.d)
                xdot4_ab, ydot4_ab = LotkaVolterra.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt, self.alpha,
                                                          self.beta, self.gamma, self.delta, self.p, self.d)

                if pp_p == 0:
                    eps_x_ab[feature_p] += self.mul
                    label_x[feature_p] += 1
                else:
                    eps_y_ab[feature_p] += self.mul
                    label_y[feature_p] += 1

                xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt6 + eps_x_ab
                ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt6 + eps_y_ab
            else:
                xdot1_ab, ydot1_ab = LotkaVolterra.f(x_ab, y_ab, self.alpha, self.beta, self.gamma, self.delta,
                                                          self.p, self.d)
                xdot2_ab, ydot2_ab = LotkaVolterra.f(x_ab + xdot1_ab * dt2, y + ydot1_ab * dt2, self.alpha,
                                                          self.beta, self.gamma, self.delta, self.p, self.d)
                xdot3_ab, ydot3_ab = LotkaVolterra.f(x_ab + xdot2_ab * dt2, y_ab + ydot2_ab * dt2, self.alpha,
                                                          self.beta, self.gamma, self.delta, self.p, self.d)
                xdot4_ab, ydot4_ab = LotkaVolterra.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt, self.alpha,
                                                          self.beta, self.gamma, self.delta, self.p, self.d)
                lst_val = [self.mul * self.mul, self.mul, self.mul]
                if pp_p == 0:
                    xnew_temp = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt6 + eps_x_ab
                    label_x[feature_p] += 1
                    for i in feature_p:
                        xdot1_ab[i] *= lst_val[seq_k]
                        xdot1_ab[i] = np.clip(xdot1_ab[i], 60000, 120000)
                    xnew_ab = x_ab + xdot1_ab * dt6 + eps_x_ab
                    ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + xdot4_ab) * dt6 + eps_y_ab
                    eps_x_ab += xnew_ab - xnew_temp
                else:
                    label_y[feature_p] += 1
                    ynew_temp = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt6 + eps_y_ab
                    for i in feature_p:
                        ydot1_ab[i] *= lst_val[seq_k]
                        ydot1_ab[i] = np.clip(ydot1_ab[i], 60000, 120000)
                    xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt6 + eps_x_ab
                    ynew_ab = y_ab + ydot1_ab * dt6 + eps_y_ab
                    eps_y_ab += ynew_ab - ynew_temp

            return (np.maximum(xnew, 0), np.maximum(ynew, 0),
                    eps_x, eps_y,
                    np.maximum(xnew_ab, 0), np.maximum(ynew_ab, 0),
                    eps_x_ab, eps_y_ab,
                    label_x, label_y)
        else:
            label_x = np.zeros(self.p)
            label_y = np.zeros(self.p)
            xdot1, ydot1 = LotkaVolterra.f(x, y, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot2, ydot2 = LotkaVolterra.f(x + xdot1 * dt2, y + ydot1 * dt2, self.alpha, self.beta, self.gamma,
                                                self.delta, self.p, self.d)
            xdot3, ydot3 = LotkaVolterra.f(x + xdot2 * dt2, y + ydot2 * dt2, self.alpha, self.beta, self.gamma,
                                                self.delta, self.p, self.d)
            xdot4, ydot4 = LotkaVolterra.f(x + xdot3 * dt, y + ydot3 * dt, self.alpha, self.beta, self.gamma,
                                                self.delta, self.p, self.d)

            eps_x = np.random.normal(scale=self.sigma, size=self.p)
            eps_y = np.random.normal(scale=self.sigma, size=self.p)
            eps_x_ab = eps_x.copy()
            eps_y_ab = eps_y.copy()
            xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt6 + eps_x
            ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt6 + eps_y

            xdot1_ab, ydot1_ab = LotkaVolterra.f(x_ab, y_ab, self.alpha, self.beta, self.gamma, self.delta, self.p,
                                                      self.d)
            xdot2_ab, ydot2_ab = LotkaVolterra.f(x_ab + xdot1_ab * dt2, y + ydot1_ab * dt2, self.alpha, self.beta,
                                                      self.gamma, self.delta, self.p, self.d)
            xdot3_ab, ydot3_ab = LotkaVolterra.f(x_ab + xdot2_ab * dt2, y_ab + ydot2_ab * dt2, self.alpha,
                                                      self.beta, self.gamma, self.delta, self.p, self.d)
            xdot4_ab, ydot4_ab = LotkaVolterra.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt, self.alpha, self.beta,
                                                      self.gamma, self.delta, self.p, self.d)

            xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt6 + eps_x_ab
            ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt6 + eps_y_ab

            return (np.maximum(xnew, 0), np.maximum(ynew, 0),
                    eps_x, eps_y,
                    np.maximum(xnew_ab, 0), np.maximum(ynew_ab, 0),
                    eps_x_ab, eps_y_ab,
                    label_x, label_y)

    def next_value(self, data, eps_norm, dt=0.01, downsample_factor=10):
        x_all = data[:, :self.p]
        y_all = data[:, self.p:]
        lst_results = []
        dt2 = dt / 2.0
        dt6 = dt / 6.0
        for k in range(len(data)):
            x = x_all[k].copy()
            y = y_all[k].copy()
            for i in range(downsample_factor):
                xdot1, ydot1 = LotkaVolterra.f(x, y, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot2, ydot2 = LotkaVolterra.f(x + xdot1 * dt2, y + ydot1 * dt2, self.alpha, self.beta, self.gamma,
                                                    self.delta, self.p, self.d)
                xdot3, ydot3 = LotkaVolterra.f(x + xdot2 * dt2, y + ydot2 * dt2, self.alpha, self.beta, self.gamma,
                                                    self.delta, self.p, self.d)
                xdot4, ydot4 = LotkaVolterra.f(x + xdot3 * dt, y + ydot3 * dt, self.alpha, self.beta, self.gamma,
                                                    self.delta, self.p, self.d)
                if i == downsample_factor - 1:
                    eps_x = eps_norm[k, :self.p]
                    eps_y = eps_norm[k, self.p:]
                else:
                    eps_x = np.zeros(self.p)
                    eps_y = np.zeros(self.p)
                xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt6 + eps_x
                ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt6 + eps_y
                x = np.maximum(xnew, 0)
                y = np.maximum(ynew, 0)
            lst_results.append(np.concatenate((x, y)))
        return np.array(lst_results)

    @staticmethod
    @njit
    def f(x, y, alpha, beta, gamma, delta, p, d):
        xdot = np.empty(p)
        ydot = np.empty(p)
        for j in range(p):
            start = ((j + d) // d) * d - d
            end = ((j + d) // d) * d
            sum_y = 0.0
            sum_x = 0.0
            for idx in range(start, end):
                sum_y += y[idx]
                sum_x += x[idx]
            # Note: 2.75 * 10e-5 equals approximately 2.75e-4.
            xdot[j] = alpha * x[j] - beta * x[j] * sum_y - 2.75e-4 * (x[j] / 200) ** 2
            ydot[j] = delta * sum_x * y[j] - gamma * y[j]
        return xdot, ydot

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list.npy'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list.npy'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct.npy'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'signed_causal_struct.npy'), self.data_dict['signed_causal_struct'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['signed_causal_struct'] = np.load(os.path.join(self.data_dir, 'signed_causal_struct.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
