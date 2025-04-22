import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
import zipfile
from datetime import datetime
import os
from functools import reduce


def download_and_extract_zenodo_msds():
    """
    Downloads and extracts the MSDS dataset from Zenodo into ./datasets/msds/.
    If the dataset is already downloaded and extracted, the function does nothing.
    """
    zenodo_url = "https://zenodo.org/api/records/3549604/files-archive"
    target_dir = os.path.join(os.getcwd(), 'datasets', 'msds')
    zip_path = os.path.join(target_dir, "msds_dataset.zip")

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Download the ZIP archive if not already downloaded
    if not os.path.exists(zip_path):
        print("Downloading dataset from Zenodo...")
        response = requests.get(zenodo_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: status code {response.status_code}")

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded ZIP file to: {zip_path}")
    else:
        print(f"ZIP file already exists at: {zip_path}. Skipping download.")

    # Extract the contents
    print("Extracting contents...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Extraction complete to: {target_dir}")

    # Extract sequential_data.zip
    sequential_zip_path = os.path.join(target_dir, "concurrent data.zip")
    extracted_flag = os.path.join(target_dir, "MSDS")  # This is the root folder in the ZIP
    if os.path.exists(sequential_zip_path):
        print("Extracting concurrent data.zip...")
        with zipfile.ZipFile(sequential_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_flag)
        print(f"Extraction of concurrent data.zip complete to: {extracted_flag}")
    else:
        print("concurrent data.zip not found.")

    print("Finished downloading and extracting the MSDS dataset.")


def preprocess_metrics_data():
    """
    Preprocesses the metrics data from the MSDS dataset.
    Reference: https://github.com/imperial-qore/TranAD/tree/main/data/MSDS
    :return:
    """
    print("Preprocessing metrics data...")
    target_dir = os.path.join(os.getcwd(), 'datasets', 'msds')
    metrics_dir = os.path.join(target_dir, 'MSDS', 'concurrent data', 'metrics')
    files = ['wally122_metrics_concurrent.csv', 'wally113_metrics_concurrent.csv', 'wally123_metrics_concurrent.csv', 'wally117_metrics_concurrent.csv', 'wally124_metrics_concurrent.csv']
    dfs = []

    # Read csv files
    for file in files:
        if '.csv' in file:
            df = pd.read_csv(os.path.join(metrics_dir, file))
            df = df.drop(columns=['load.cpucore', 'load.min1', 'load.min5', 'load.min15'])
            dfs.append(df)

    # Process dataframes
    start = dfs[0].min()['now']
    end = dfs[0].max()['now']
    for df in dfs:
        if df.min()['now'] > start:
            start = df.min()['now']
        if df.max()['now'] < end:
            end = df.max()['now']
    id_vars = ['now']
    dfs2 = []
    for df in dfs:
        df = df.drop(np.argwhere(list(df['now'] < start)).reshape(-1))
        df = df.drop(np.argwhere(list(df['now'] > end)).reshape(-1))
        melted = df.melt(id_vars=id_vars).dropna()
        df = melted.pivot_table(index=id_vars, columns="variable", values="value")
        dfs2.append(df)
    dfs = dfs2

    dfs_unique = []
    for idx, df in enumerate(dfs):
        df = df.copy()
        # Rename all columns except the index (assumed to be 'now')
        df.columns = [f"{col}_{idx}" if col != "now" else col for col in df.columns]
        dfs_unique.append(df)

    df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs_unique)

    # Change timezone string format
    ni = []
    for i in df_merged.index:
        dt = datetime.strptime(i[:-5], '%Y-%m-%d %H:%M:%S')
        ni.append(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
    df_merged.index = ni

    # Save train and test sets
    start = round(df_merged.shape[0] * 0.1)
    df_merged = df_merged[start:]
    split = round(df_merged.shape[0] / 2)
    df_merged[:split].to_csv(os.path.join(target_dir, 'train.csv'))
    df_merged[split:].to_csv(os.path.join(target_dir, 'test.csv'))
    print("Preprocessing complete. Train and test sets saved.")


class MSDS:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']

    def generate_example(self):
        # Ensure the dataset is downloaded and extracted
        download_and_extract_zenodo_msds()
        preprocess_metrics_data()
        # load data and save to csv
        df_label = pd.read_csv(os.path.join(self.data_dir, 'labels.csv'))
        df_normal = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        df_abnormal = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

        df_normal, df_abnormal = df_normal.values[::5, 1:], df_abnormal.values[::5, 1:]
        df_label = df_label.values[::5, 1:]
        labels = np.max(df_label, axis=1)

        x_n_list = []
        for i in range(0, len(df_normal), 10000):
            if i + 10000 < len(df_normal):
                x_n_list.append(df_normal[i:i + 10000])
        test_x_lst = []
        label_lst = []

        for i in np.where(labels == 1)[0]:
            if i - 2 * self.window_size > 0 and i + self.window_size < len(df_abnormal):
                if sum(labels[i - 2 * self.window_size:i]) == 0:
                    test_x_lst.append(df_abnormal[i-2*self.window_size:i + self.window_size])
                    label_lst.append(df_label[i-2*self.window_size:i + self.window_size])

        scaler = MinMaxScaler()
        scaler.fit(np.concatenate(x_n_list, axis=0))
        x_n_list = [scaler.transform(i) for i in x_n_list]
        test_x_lst = [scaler.transform(i) for i in test_x_lst]
        self.data_dict['x_n_list'] = np.array(x_n_list)
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(label_lst)

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=False)
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=True)
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=True)
