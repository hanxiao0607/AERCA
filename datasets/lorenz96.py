import numpy as np
from scipy.integrate import ode
from tqdm import tqdm
import os

class Lorenz96:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.mul = options['mul']
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.downsample_factor = options['downsample_factor']
        self.system_ode = ode(self._system_ode).set_integrator('vode', method='bdf')
        self.force = options['force']
        self.dependent_features = options['dependent_features']

    @staticmethod
    def _system_ode(t: float, x: np.array, theta: np.array) -> list:
        """
        Computes the Lorenz96 derivatives using vectorized operations.
        """
        force = theta[1]
        # Flatten the state vector and determine its dimension
        x = np.asarray(x).flatten()
        # Compute the derivative using vectorized cyclic shifts.
        f = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + force
        return f.tolist()

    def gen_data_point(self, downsample=True):
        """
        Integrate the Lorenz96 system using an ODE solver and inject anomalies.
        Downsampling and anomaly injection are handled as in the original code,
        but with improvements in accumulation and anomaly detection.
        """
        # Integration parameters
        t_start = 0.0
        t_delta_integration = 0.01
        t_end = 10 * (self.t - 1) * t_delta_integration

        # Generate initial states
        x = list(np.random.rand(self.num_vars, 1))
        # other
        x_n = np.copy(x).reshape(self.num_vars, 1)
        t = [t_start]
        # initialize LV ODE system
        self.system_ode.set_initial_value(x, t_start).set_f_params([self.num_vars, self.force])
        # integrate until specified
        while self.system_ode.successful() and self.system_ode.t < t_end:
            self.system_ode.integrate(t=self.system_ode.t + t_delta_integration)
            x_n = np.c_[x_n, self.system_ode.y.reshape(self.num_vars, 1)]
            t.append(self.system_ode.t)
        # Downsample the time series
        if downsample:
            x_n = x_n[:, ::self.downsample_factor]
        x_ab = np.copy(x)
        lst_label = np.zeros((self.t*self.downsample_factor, self.num_vars))
        t_ab = np.random.randint(int(0.5*self.t), int(0.8*self.t), size=1)
        if self.adlength > 1:
            temp_t_ab = []
            for i in range(self.adlength):
                temp_t_ab.append(t_ab+i)
            t_ab = np.array(temp_t_ab)
        feature_ab = np.random.permutation(np.arange(self.num_vars))[:np.random.randint(1, self.num_vars//2+1)]
        lst_label[t_ab*self.downsample_factor, feature_ab] = 1
        self.system_ode.set_initial_value(x_ab, t_start).set_f_params([self.num_vars, self.force])
        # integrate until specified
        while self.system_ode.successful() and self.system_ode.t < t_end:
            self.system_ode.integrate(t=self.system_ode.t + t_delta_integration)
            if int((1/t_delta_integration)*(self.system_ode.t)) in t_ab*self.downsample_factor:
                if self.adtype == 'non_causal':
                    assert self.system_ode.y.reshape(-1,1).shape == lst_label[int(self.system_ode.t), :].reshape(-1,1).shape
                    x_ab = np.c_[x_ab, self.system_ode.y.reshape(self.num_vars, 1) + self.mul *
                                       np.array((lst_label[int((1/t_delta_integration)*(self.system_ode.t)), :]).reshape(self.num_vars, 1))]
                    self.system_ode.set_initial_value(x_ab[:, -1], self.system_ode.t)
                elif self.adtype == 'causal':
                    raise NotImplementedError("Causal anomaly not implemented for this dataset.")
                else:
                    raise NotImplementedError("Invalid adtype. Expected 'non_causal' or 'causal'.")
            else:
                x_ab = np.c_[x_ab, self.system_ode.y.reshape(self.num_vars, 1)]
        x_ab = x_ab[:, ::self.downsample_factor]
        lst_label = lst_label[::self.downsample_factor, :]
        assert lst_label.shape == np.transpose(x_n).shape
        assert lst_label.shape == np.transpose(x_ab).shape, print(lst_label.shape, np.transpose(x_ab).shape)
        return np.transpose(x_n), np.transpose(x_ab), lst_label

    def get_causal_structure(self):
        """
        Generates a causal structure matrix for the Lorenz96 system.
        """
        a = np.zeros((self.num_vars, self.num_vars))
        for i in range(self.num_vars):
            a[i, i] = 1
            a[(i + 1) % self.num_vars, i] = 1
            a[(i + 2) % self.num_vars, i] = 1
            a[(i - 1) % self.num_vars, i] = 1
        return a

    def generate_example(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        for _ in tqdm(range(self.n), desc='Generating data'):
            x_n, x_ab, label = self.gen_data_point()
            x_n_list.append(x_n)
            x_ab_list.append(x_ab)
            eps_n_list.append(np.zeros((self.t, self.num_vars)))
            eps_ab_list.append(np.zeros((self.t, self.num_vars)))
            label_list.append(label)

        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['eps_n_list'] = np.array(eps_n_list)
        self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
        self.data_dict['label_list'] = np.array(label_list)
        self.data_dict['causal_struct'] = self.get_causal_structure()
        self.data_dict['signed_causal_struct'] = []

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
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])


    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['signed_causal_struct'] = None
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))

