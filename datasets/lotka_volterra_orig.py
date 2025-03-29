# Based on https://github.com/smkalami/lotka-volterra-in-python
import numpy as np
from tqdm import tqdm
import os
from numba import jit, njit

class MultiLotkaVolterra:
    def __init__(self, options):
        """
        Dynamical multi-species Lotka--Volterra system. The original two-species Lotka--Volterra is a special case
        with p = 1 , d = 1.
        @param p: number of predator/prey species. Total number of variables is 2*p.
        @param d: number of GC parents per variable.
        @param alpha: strength of interaction of a prey species with itself.
        @param beta: strength of predator -> prey interaction.
        @param gamma: strength of interaction of a predator species with itself.
        @param delta: strength of prey -> predator interaction.
        @param sigma: scale parameter for the noise.
        """
        self.options = options
        self.data_dict = {}
        self.p = options['num_vars']//2
        self.d = options['d']
        self.dt = options['dt']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.seed = options['seed']
        self.downsample_factor = options['downsample_factor']
        self.data_dir = options['data_dir']
        self.mul = options['mul']

        # assert self.p >= self.d and self.p % self.d == 0

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

        # Initialize lists to store results
        lst_n = []
        lst_ab = []
        eps_n = []
        eps_ab = []
        lst_labels = []
        for _ in tqdm(range(self.n)):
            xs_0 = np.random.uniform(10, 20, size=(self.p, ))
            ys_0 = np.random.uniform(10, 20, size=(self.p, ))

            ts = np.arange(self.t) * self.dt

            # Simulation Loop
            xs = np.zeros((self.t, self.p))
            ys = np.zeros((self.t, self.p))
            eps_x = np.zeros((self.t, self.p))
            eps_y = np.zeros((self.t, self.p))
            xs[0, :] = xs_0
            ys[0, :] = ys_0

            xs_ab = np.zeros((self.t, self.p))
            ys_ab = np.zeros((self.t, self.p))
            eps_x_ab = np.zeros((self.t, self.p))
            eps_y_ab = np.zeros((self.t, self.p))
            label_x = np.zeros((self.t, self.p))
            label_y = np.zeros((self.t, self.p))
            xs_ab[0, :] = xs_0
            ys_ab[0, :] = ys_0
            t_p = np.random.randint((0.5*self.t)//self.downsample_factor, self.t//self.downsample_factor, size=1)
            if self.adlength > 1:
                temp_t_p = []
                for i in range(self.adlength):
                    temp_t_p.append(t_p + i)
                t_p = np.array(temp_t_p)
            pp_p = np.random.randint(0, 2, size=1)
            try:
                feature_p = np.random.permutation(np.arange(self.p))[:np.random.randint(3, min(5, self.p)+1)]
            except:
                feature_p = np.random.permutation(np.arange(self.p))[:np.random.randint(2, min(5, self.p) + 1)]
            count = 0
            for k in range(self.t - 1):
                if k in (t_p*self.downsample_factor)-1:
                    xs[k + 1, :], ys[k + 1, :], eps_x[k + 1, :], eps_y[k + 1, :], xs_ab[k + 1, :], ys_ab[k + 1, :], \
                    eps_x_ab[k + 1, :], eps_y_ab[k + 1, :], label_x[k + 1, :], label_y[k + 1, :] = self.next(xs[k, :], ys[k, :], xs_ab[k, :], ys_ab[k, :],
                                                                       self.dt, ab=1, pp_p=pp_p, feature_p=feature_p, adtype=self.adtype, seq_k=count)
                    count += 1
                else:
                    xs[k + 1, :], ys[k + 1, :], eps_x[k + 1, :], eps_y[k + 1, :], xs_ab[k + 1, :], ys_ab[k + 1, :], \
                    eps_x_ab[k + 1, :], eps_y_ab[k + 1, :], label_x[k + 1, :], label_y[k + 1, :] = self.next(xs[k, :], ys[k, :], xs_ab[k, :], ys_ab[k, :], self.dt)

            lst_n.extend([np.concatenate((xs[::self.downsample_factor, :], ys[::self.downsample_factor, :]), 1)])
            eps_n.extend([np.concatenate((eps_x[::self.downsample_factor, :], eps_y[::self.downsample_factor, :]), 1)])
            lst_ab.extend([np.concatenate((xs_ab[::self.downsample_factor, :], ys_ab[::self.downsample_factor, :]), 1)])
            eps_ab.extend([np.concatenate((eps_x_ab[::self.downsample_factor, :], eps_y_ab[::self.downsample_factor, :]), 1)])
            lst_labels.extend([np.concatenate((label_x[::self.downsample_factor, :], label_y[::self.downsample_factor, :]), 1)])
        causal_struct = np.zeros((self.p * 2, self.p * 2))
        signed_causal_struct = np.zeros((self.p * 2, self.p * 2))
        for j in range(self.p):
            # Self causation
            causal_struct[j, j] = 1
            causal_struct[j + self.p, j + self.p] = 1

            signed_causal_struct[j, j] = +1
            signed_causal_struct[j + self.p, j + self.p] = -1

            # Predator-prey relationships
            causal_struct[j, int(np.floor((j + self.d) / self.d) * self.d - 1 + self.p - self.d + 1):int(np.floor((j + self.d) / self.d) * self.d + self.p)] = 1
            causal_struct[j + self.p, int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) / self.d) * self.d)] = 1

            signed_causal_struct[j, int(np.floor((j + self.d) / self.d) * self.d - 1 + self.p - self.d + 1):int(np.floor((j + self.d) / self.d) * self.d + self.p)] = -1
            signed_causal_struct[j + self.p, int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) / self.d) * self.d)] = +1

        data_dict = {}
        data_dict['x_n_list'] = np.array(lst_n)[:, 50:, :]
        data_dict['eps_n_list'] = np.array(eps_n)[:, 50:, :]
        data_dict['x_ab_list'] = np.array(lst_ab)[:, 50:, :]
        data_dict['eps_ab_list'] = np.array(eps_ab)[:, 50:, :]
        data_dict['label_list'] = np.array(lst_labels)[:, 50:]
        data_dict['causal_struct'] = causal_struct
        data_dict['signed_causal_struct'] = signed_causal_struct
        self.data_dict = data_dict

    # Dynamics
    # State transitions using the Runge-Kutta method
    def next(self, x, y, x_ab, y_ab, dt, ab=0, pp_p=0, feature_p=None, adtype='non_causal', seq_k=0):
        if ab == 1:
            label_x = np.zeros((self.p,))
            label_y = np.zeros((self.p,))
            xdot1, ydot1 = MultiLotkaVolterra.f(x, y, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot2, ydot2 = MultiLotkaVolterra.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot3, ydot3 = MultiLotkaVolterra.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot4, ydot4 = MultiLotkaVolterra.f(x + xdot3 * dt, y + ydot3 * dt, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            # Add noise to simulations
            eps_x = np.random.normal(scale=self.sigma, size=(self.p,))
            eps_y = np.random.normal(scale=self.sigma, size=(self.p,))
            eps_x_ab = eps_x.copy()
            eps_y_ab = eps_y.copy()
            xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + \
                   eps_x
            ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + \
                   eps_y
            if adtype == 'non_causal':
                xdot1_ab, ydot1_ab = MultiLotkaVolterra.f(x_ab, y_ab, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot2_ab, ydot2_ab = MultiLotkaVolterra.f(x_ab + xdot1_ab * dt / 2, y + ydot1_ab * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot3_ab, ydot3_ab = MultiLotkaVolterra.f(x_ab + xdot2_ab * dt / 2, y_ab + ydot2_ab * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot4_ab, ydot4_ab = MultiLotkaVolterra.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)

                # Add noise to simulations
                if pp_p == 0:
                    eps_x_ab[feature_p] += self.mul
                    label_x[feature_p] += 1
                else:
                    eps_y_ab[feature_p] += self.mul
                    label_y[feature_p] += 1

                xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt / 6 + \
                          eps_x_ab
                ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt / 6 + \
                          eps_y_ab
            else:
                xdot1_ab, ydot1_ab = MultiLotkaVolterra.f(x_ab, y_ab, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot2_ab, ydot2_ab = MultiLotkaVolterra.f(x_ab + xdot1_ab * dt / 2, y + ydot1_ab * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot3_ab, ydot3_ab = MultiLotkaVolterra.f(x_ab + xdot2_ab * dt / 2, y_ab + ydot2_ab * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot4_ab, ydot4_ab = MultiLotkaVolterra.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                lst_val = [self.mul*self.mul, self.mul, self.mul]
                if pp_p == 0:
                    xnew_temp = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt / 6 + \
                                eps_x_ab
                    label_x[feature_p] += 1
                    for i in feature_p:
                        xdot1_ab[i] = xdot1_ab[i]*lst_val[seq_k]
                        if xdot1_ab[i] > 120000:
                            xdot1_ab[i] = 120000
                        if xdot1_ab[i] < 60000:
                            xdot1_ab[i] = 60000
                    xnew_ab = x_ab + xdot1_ab * dt / 6 + \
                              eps_x_ab
                    ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt / 6 + \
                              eps_y_ab
                    eps_x_ab = eps_x_ab + xnew_ab - xnew_temp
                else:
                    label_y[feature_p] += 1
                    ynew_temp = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt / 6 + \
                                eps_y_ab
                    for i in feature_p:
                        ydot1_ab[i] = ydot1_ab[i]*lst_val[seq_k]
                        if ydot1_ab[i] > 120000:
                            ydot1_ab[i] = 120000
                        if ydot1_ab[i] < 60000:
                            ydot1_ab[i] = 60000
                    xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt / 6 + \
                              eps_x_ab
                    ynew_ab = y_ab + ydot1_ab * dt / 6 + \
                              eps_y_ab
                    eps_y_ab = eps_y_ab + ynew_ab - ynew_temp
            # Clip from below to prevent populations from becoming negative
        else:
            label_x = np.zeros((self.p,))
            label_y = np.zeros((self.p,))
            xdot1, ydot1 = MultiLotkaVolterra.f(x, y, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot2, ydot2 = MultiLotkaVolterra.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot3, ydot3 = MultiLotkaVolterra.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot4, ydot4 = MultiLotkaVolterra.f(x + xdot3 * dt, y + ydot3 * dt, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            # Add noise to simulations
            eps_x = np.random.normal(scale=self.sigma, size=(self.p, ))
            eps_y = np.random.normal(scale=self.sigma, size=(self.p, ))
            eps_x_ab = eps_x.copy()
            eps_y_ab = eps_y.copy()
            xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + \
                   eps_x
            ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + \
                   eps_y

            xdot1_ab, ydot1_ab = MultiLotkaVolterra.f(x_ab, y_ab, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot2_ab, ydot2_ab = MultiLotkaVolterra.f(x_ab + xdot1_ab * dt / 2, y + ydot1_ab * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot3_ab, ydot3_ab = MultiLotkaVolterra.f(x_ab + xdot2_ab * dt / 2, y_ab + ydot2_ab * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            xdot4_ab, ydot4_ab = MultiLotkaVolterra.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
            # Add noise to simulations
            xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt / 6 + \
                      eps_x_ab
            ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt / 6 + \
                      eps_y_ab
            # Clip from below to prevent populations from becoming negative
        return np.maximum(xnew, 0), np.maximum(ynew, 0), eps_x, eps_y, np.maximum(xnew_ab, 0), np.maximum(ynew_ab, 0), \
               eps_x_ab, eps_y_ab, label_x, label_y

    def next_value(self, data, eps_norm, dt=0.01, downsample_factor=10):
        x_all = data[:, :self.p]
        y_all = data[:, self.p:]
        lst_results = []
        for k in range(len(data)):
            x = x_all[k]
            y = y_all[k]
            for i in range(downsample_factor):
                xdot1, ydot1 = MultiLotkaVolterra.f(x, y, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot2, ydot2 = MultiLotkaVolterra.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot3, ydot3 = MultiLotkaVolterra.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                xdot4, ydot4 = MultiLotkaVolterra.f(x + xdot3 * dt, y + ydot3 * dt, self.alpha, self.beta, self.gamma, self.delta, self.p, self.d)
                # Add noise to simulations
                if i == downsample_factor-1:
                    eps_x = eps_norm[k, :self.p]
                    eps_y = eps_norm[k, self.p:]
                else:
                    eps_x = np.zeros((self.p,))
                    eps_y = np.zeros((self.p,))
                xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + \
                       eps_x
                ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + \
                       eps_y
                x = np.maximum(xnew, 0).copy()
                y = np.maximum(ynew, 0).copy()
            lst_results.append(np.concatenate((x, y)))
        return np.array(lst_results)

    @staticmethod
    @njit
    def f(x, y, alpha, beta, gamma, delta, p, d):
        xdot = np.zeros((p,))
        ydot = np.zeros((p,))

        for j in range(p):
            y_Nxj = y[int(np.floor((j + d) / d) * d - d + 1 - 1):int(np.floor((j + d) / d) * d)]
            x_Nyj = x[int(np.floor((j + d) / d) * d - d + 1 - 1):int(np.floor((j + d) / d) * d)]
            xdot[j] = alpha * x[j] - beta * x[j] * np.sum(y_Nxj) - 2.75 * 10e-5 * (x[j] / 200) ** 2
            ydot[j] = delta * np.sum(x_Nyj) * y[j] - gamma * y[j]
        return xdot, ydot

    def save_data(self):
        # Create the directory if it does not exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Save the data
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



