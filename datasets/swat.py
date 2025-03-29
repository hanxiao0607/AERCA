import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class SWaT:
    def __init__(self, options):
        """
        Initialize the SWaT dataset processing class with the given options.

        Parameters:
        - options (dict): A dictionary containing keys such as 'seed', 'num_vars',
                          'data_dir', 'window_size', and 'shuffle'.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']

    def generate_example(self):
        """
        Generate examples by loading, cleaning, and processing the SWaT dataset.
        This method loads the label, normal, and abnormal data files, performs
        necessary cleaning, scaling, and window slicing operations, and stores
        the processed arrays in self.data_dict.
        """
        # ----------------------------
        # Load Attack Label Data
        # ----------------------------
        label_file = os.path.join(self.data_dir, 'List_of_attacks_Final.xlsx')
        df_label = pd.read_excel(label_file, header=0, index_col=0)


        # ----------------------------
        # Load Normal and Abnormal Data
        # ----------------------------
        normal_csv = os.path.join(self.data_dir, 'SWaT_Normal.csv')
        abnormal_csv = os.path.join(self.data_dir, 'SWaT_Abnormal.csv')

        if os.path.exists(normal_csv) and os.path.exists(abnormal_csv):
            df_normal = pd.read_csv(normal_csv, header=0, index_col=0)
            df_abnormal = pd.read_csv(abnormal_csv, header=0, index_col=0)
        else:
            normal_excel = os.path.join(self.data_dir, 'SWaT_Dataset_Normal_v1.xlsx')
            abnormal_excel = os.path.join(self.data_dir, 'SWaT_Dataset_Attack_v0.xlsx')
            df_normal = pd.read_excel(normal_excel, header=1)
            df_normal.to_csv(normal_csv)
            df_abnormal = pd.read_excel(abnormal_excel, header=1)
            df_abnormal.to_csv(abnormal_csv)

        # ----------------------------
        # Clean Label Data
        # ----------------------------
        # Drop rows where 'Start Time' or 'End Time' is missing
        df_label_clean = df_label.dropna(subset=['Start Time', 'End Time'], how='any').copy()
        # Remove columns not needed for further processing
        df_label_clean.drop(columns=['Start State', 'Attack', 'Expected Impact or attacker intent',
                                      'Unexpected Outcome', 'Actual Change'], inplace=True)
        # Convert 'Start Time' and 'End Time' to datetime for processing
        df_label_clean['Start Time'] = pd.to_datetime(df_label_clean['Start Time'])
        # Construct 'Adjusted End Time' by combining the date from 'Start Time' with the time from 'End Time'
        df_label_clean['Adjusted End Time'] = df_label_clean.apply(
            lambda row: pd.to_datetime(
            row['Start Time'].strftime('%Y-%m-%d') + ' ' + row['End Time'].strftime('%H:%M:%S')), axis=1)
        # Save cleaned label data to CSV
        df_label_clean.to_csv(os.path.join(self.data_dir, 'SWaT_label.csv'))

        # ----------------------------
        # Clean Normal Data
        # ----------------------------
        # Select only rows marked as 'Normal'
        df_normal = df_normal.loc[df_normal['Normal/Attack'] == 'Normal']
        # Drop unnecessary columns and downsample by taking every 10th row
        df_normal.drop(columns=[' Timestamp', 'Normal/Attack'], inplace=True)
        df_normal = df_normal[::10].reset_index(drop=True)

        # ----------------------------
        # Clean Abnormal Data
        # ----------------------------
        # Remove any rows with missing values and reset index
        df_abnormal.dropna(how='any', inplace=True)
        df_abnormal.reset_index(drop=True, inplace=True)
        # Initialize label matrix with zeros; columns from 1 to second-last column are used
        labels = np.zeros(df_abnormal.values[:, 1:-1].shape)
        # Convert the timestamp column to datetime using the given format, then standardize its format
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(
            df_abnormal[' Timestamp'], format=' %d/%m/%Y %I:%M:%S %p'
        ).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(df_abnormal['Adjusted Timestamp'])

        # ----------------------------
        # Create Column Dictionary for Abnormal Data
        # ----------------------------
        # Create a mapping from cleaned column names (without leading spaces) to their index
        col_dic = {}
        for i in df_abnormal.columns.values[1:-2]:
            col_dic[i.lstrip()] = len(col_dic)

        # ----------------------------
        # Process Each Attack Event for Abnormal Data
        # ----------------------------
        test_x_lst = []
        test_label_lst = []

        for i in range(len(df_label_clean)):
            # Define the lower and upper time bounds for the attack event
            lower = df_label_clean.iloc[i]['Start Time']
            upper = df_label_clean.iloc[i]['Adjusted End Time']
            # Extract the list of attack points (column names) from the label data
            attack_lst = df_label_clean.iloc[i]['Attack Point'].split(",")
            # Map attack points to their corresponding column indices
            attack_lst_ind = [col_dic[j.replace('-', '').lstrip().upper()] for j in attack_lst]
            # Find indices in abnormal data where the timestamp is within the attack interval and marked as 'Attack'
            index_lst = np.array(df_abnormal.loc[
                (df_abnormal['Adjusted Timestamp'] >= lower) &
                (df_abnormal['Adjusted Timestamp'] <= upper) &
                (df_abnormal['Normal/Attack'] == 'Attack')
            ].index.values)
            if len(index_lst) > 0:
                # Mark the corresponding attack points in the label matrix as 1 for these indices
                for j in attack_lst_ind:
                    labels[index_lst, j] = 1
                # Define the window for the example based on the minimum index in the attack interval
                start_idx = int(min(index_lst) - 2 * 10 * self.window_size)
                end_idx = int(min(index_lst) + 1 * 10 * self.window_size)
                # Slice the abnormal data and label arrays with a step of 10
                test_x_lst.append(
                    df_abnormal.iloc[start_idx:end_idx:10, 1:-2].values
                )
                test_label_lst.append(
                    labels[start_idx:end_idx:10]
                )

        # ----------------------------
        # Process Normal Data: Split and Scale
        # ----------------------------
        # Split normal data into segments of 1000 rows each, ensuring each segment has exactly 1000 rows
        x_n_list = [
            df_normal.iloc[i:i + 1000].values
            for i in range(0, len(df_normal), 1000)
            if i + 1000 < len(df_normal)
        ]
        # Initialize and fit the StandardScaler on the concatenated normal data segments
        scaler = StandardScaler()
        scaler.fit(np.concatenate(x_n_list, axis=0))
        # Transform each segment of normal data
        x_n_list = [scaler.transform(segment) for segment in x_n_list]
        # Transform each abnormal example using the same scaler
        test_x_lst = [scaler.transform(example) for example in test_x_lst]

        # ----------------------------
        # Store Processed Data in data_dict
        # ----------------------------
        self.data_dict['x_n_list'] = np.array(x_n_list)
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(test_label_lst)

    def save_data(self):
        """
        Save the processed data arrays to .npy files in the data directory.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        """
        Load the processed data arrays from .npy files in the data directory into data_dict.
        """
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=False)
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=True)
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=True)
