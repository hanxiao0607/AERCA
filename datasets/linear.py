# Some synthetic datasets with linear dynamics
import numpy as np
import os
import random


class Linear:
    def __init__(self, options):
        """
        Initializes the LinearDynamics class to generate synthetic datasets with linear dynamics.

        Args:
            options (dict): Configuration options for the dataset generation.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.mul = options['mul']
        self.a = options['a'] if options['a'] is not None else self._generate_random_coefficients()
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.data_dir = options['data_dir']
        self.dependent_features = options['dependent_features']

    def _generate_random_coefficients(self):
        """
        Generates random coefficients for the linear dynamics.

        Returns:
            np.ndarray: Array of random coefficients.
        """
        a = np.zeros((8,))
        for k in range(8):
            u_1 = np.random.uniform(0, 1, 1)
            a[k] = np.random.uniform(-0.8, -0.2, 1) if u_1 <= 0.5 else np.random.uniform(0.2, 0.8, 1)
        return a
    def generate_example(self):
        """
        Generates synthetic time series data with linear dynamics and anomalies.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Preallocate arrays
        x_n_list = np.zeros((self.n, self.t, 4))
        x_ab_list = np.zeros((self.n, self.t, 4))
        eps_n_list = np.zeros((self.n, self.t, 4))
        eps_ab_list = np.zeros((self.n, self.t, 4))
        label_list = np.zeros((self.n, self.t, 4))

        # Generate data for each sample
        for i in range(self.n):
            # Generate random noise for each variable
            if self.dependent_features == 1:
                eps = np.random.multivariate_normal(
                    mean=np.zeros(4),
                    cov=np.array([
                        [1.0, 0.8, 0.5, 0.2],
                        [0.8, 1.0, 0.4, 0.3],
                        [0.5, 0.4, 1.0, 0.6],
                        [0.2, 0.3, 0.6, 1.0],
                    ]),
                    size=self.t
                ) * 0.4
            else:
                eps = 0.4 * np.random.randn(self.t, 4)

            # Initialize variables
            x, w, y, z = np.zeros(self.t), np.zeros(self.t), np.zeros(self.t), np.zeros(self.t)
            for j in range(1, self.t):
                x[j] = self.a[0] * x[j - 1] + eps[j, 0]
                w[j] = self.a[1] * w[j - 1] + self.a[2] * x[j - 1] + eps[j, 1]
                y[j] = self.a[3] * y[j - 1] + self.a[4] * w[j - 1] + eps[j, 2]
                z[j] = self.a[5] * z[j - 1] + self.a[6] * w[j - 1] + self.a[7] * y[j - 1] + eps[j, 3]

            # Store results
            x_n_list[i] = np.stack((x, w, y, z), axis=-1)
            eps_n_list[i] = eps

            # Generate anomalies
            start = np.random.randint(int(0.2 * self.t), int(0.8 * self.t - self.adlength))
            t_p = np.arange(start, start + self.adlength)

            feature_p = np.random.choice(4, size=np.random.randint(1, 5), replace=False)
            ab = np.zeros(4)
            ab[feature_p] += self.mul

            temp_label = np.zeros((self.t, 4))
            temp_label[np.ix_(t_p, feature_p)] = 1

            eps_ab = eps.copy()
            if self.adtype == 'non_causal':
                eps_ab[t_p] += ab

            b = self.a * 3 if self.adtype == 'causal' else None

            x, w, y, z = np.zeros(self.t), np.zeros(self.t), np.zeros(self.t), np.zeros(self.t)
            is_anomaly = np.zeros(self.t, dtype=bool)
            is_anomaly[t_p] = True

            # Generate anomaly time series data
            for j in range(1, self.t):
                if is_anomaly[j]:
                    if self.adtype == 'causal':
                        z[j] = b[0] * z[j - 1] + eps[j, 3]
                        y[j] = b[1] * y[j - 1] + b[2] * x[j - 1] + eps[j, 2]
                        w[j] = b[3] * w[j - 1] + b[4] * w[j - 1] + eps[j, 1]
                        x[j] = b[5] * x[j - 1] + b[6] * w[j - 1] + b[7] * y[j - 1] + eps[j, 0]
                    else:
                        x[j] = self.a[0] * x[j - 1] + eps_ab[j, 0]
                        w[j] = self.a[1] * w[j - 1] + self.a[2] * x[j - 1] + eps_ab[j, 1]
                        y[j] = self.a[3] * y[j - 1] + self.a[4] * w[j - 1] + eps_ab[j, 2]
                        z[j] = self.a[5] * z[j - 1] + self.a[6] * w[j - 1] + self.a[7] * y[j - 1] + eps_ab[j, 3]
                else:
                    x[j] = self.a[0] * x[j - 1] + eps[j, 0]
                    w[j] = self.a[1] * w[j - 1] + self.a[2] * x[j - 1] + eps[j, 1]
                    y[j] = self.a[3] * y[j - 1] + self.a[4] * w[j - 1] + eps[j, 2]
                    z[j] = self.a[5] * z[j - 1] + self.a[6] * w[j - 1] + self.a[7] * y[j - 1] + eps[j, 3]

            x_ab_list[i] = np.stack((x, w, y, z), axis=-1)
            eps_ab_list[i] = eps_ab
            label_list[i] = temp_label

        self.data_dict = {
            'x_n_list': x_n_list,
            'x_ab_list': x_ab_list,
            'eps_n_list': eps_n_list,
            'eps_ab_list': eps_ab_list,
            'label_list': label_list,
            'a': self.a,
            'causal_struct': np.array([[1, 0, 0, 0],
                                  [1, 1, 0, 0],
                                  [0, 1, 1, 0],
                                  [0, 1, 1, 1]]),
            'causal_struct_value': np.array([
                [self.a[0], 0, 0, 0],
                [self.a[2], self.a[1], 0, 0],
                [0, self.a[4], self.a[3], 0],
                [0, self.a[6], self.a[7], self.a[5]]
            ]),
            'signed_causal_struct': np.sign(np.array([
                [self.a[0], 0, 0, 0],
                [self.a[2], self.a[1], 0, 0],
                [0, self.a[4], self.a[3], 0],
                [0, self.a[6], self.a[7], self.a[5]]
            ]))
        }

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list.npy'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list.npy'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct.npy'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])
        np.save(os.path.join(self.data_dir, 'a.npy'), self.data_dict['a'])
        np.save(os.path.join(self.data_dir, 'causal_struct_value.npy'), self.data_dict['causal_struct_value'])
        np.save(os.path.join(self.data_dir, 'signed_causal_struct.npy'), self.data_dict['signed_causal_struct'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.data_dict['a'] = np.load(os.path.join(self.data_dir, 'a.npy'))
        self.data_dict['causal_struct_value'] = np.load(os.path.join(self.data_dir, 'causal_struct_value.npy'))
        self.data_dict['signed_causal_struct'] = np.load(os.path.join(self.data_dir, 'signed_causal_struct.npy'))

