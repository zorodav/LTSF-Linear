from pathlib import Path
import os
import yaml
import scipy.io
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from torch.utils.data import Dataset, DataLoader
import warnings

top_dir = Path(__file__).parent

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class ODE_Lorenz(Dataset):
    def __init__(self, root_path, flag='train', size=None, pair_id=3,
                 features='M', scale=True, timeenc=0, freq='h', train_only=False):
        """
        Dataset class for the ODE_Lorenz dynamical system.
        
        Parameters:
            root_path: Parent directory containing the ODE_Lorenz folder,
            flag: 'train', 'test', or 'pred'. (For simplicity, 'val' is omitted.)
            size: List or tuple [seq_len, label_len, pred_len]. Defaults are used if None.
            pair_id: The experiment pair id to load from the YAML file.
            features: Type indicator, e.g. 'M' for multivariate.
            scale: Whether to scale the data (using StandardScaler).
            timeenc: Either 0 (basic time features) or 1 (advanced encoding).
            freq: Frequency string for time features (e.g., 'h' for hourly).
            train_only: If True, do not split the data.
        """
        # Default sequence lengths if not provided.
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.flag = flag
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.pair_id = pair_id
        self.features = features

        # Define the ODE_Lorenz data folder and YAML configuration file.
        self.data_folder = top_dir.parent.parent.parent / 'data' / 'ODE_Lorenz'
        self.yaml_path = top_dir.parent.parent.parent / 'data' / 'ODE_Lorenz' / 'ODE_Lorenz.yaml'

        self.__load_config__()
        self.__read_data__()

    def __load_config__(self):
        with open(self.yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # Locate the chosen pair configuration by matching the pair id from YAML.
        self.pair_config = None
        for pair in self.config.get('pairs', []):
            if pair.get('id') == self.pair_id:
                self.pair_config = pair
                break
        if self.pair_config is None:
            raise ValueError(f"Pair id {self.pair_id} not found in YAML config.")

    def __read_data__(self):
        # Determine the subfolder name based on flag.
        folder_flag = 'train' if self.flag == 'train' else 'test'
        # For training, YAML lists files under "train" (as a list) 
        # and for testing/prediction there is a single file.
        if self.flag == 'train':
            files = self.pair_config.get('train', [])
            print(f"train_files: {files}")
            if not isinstance(files, list):
                files = [files]
        else:
            test_file = self.pair_config.get('test', None)
            print(f"test_file: {test_file}")
            if test_file is None:
                raise ValueError("Test file not specified in YAML config.")
            files = [test_file]

        data_list = []
        # Load each MAT file. We assume the MAT files are stored under the subfolder (train or test).
        for file_name in files:
            file_path = os.path.join(self.data_folder, folder_flag, file_name)
            mat = scipy.io.loadmat(file_path)
            # Use key 'data' if available; otherwise pick the first available key.
            if 'data' in mat:
                raw = mat['data']
            else:
                keys = [k for k in mat.keys() if not k.startswith('__')]
                if not keys:
                    raise ValueError(f"No valid data in MAT file: {file_path}")
                raw = mat[keys[0]]
            raw = raw.astype(np.float32)
            data_list.append(raw)
        # Concatenate if more than one MAT file is specified (assumes concatenation along the time axis).
        data = np.concatenate(data_list, axis=1) if len(data_list) > 1 else data_list[0]
        # According to YAML metadata the matrix shape is [channels, time], so we transpose to (time, channels).
        data = data.T  # now shape: (time, channels)

        # Scaling:
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data

        total_len = data_scaled.shape[0]
        # Create time features using a date range.
        date_range = pd.date_range(start='2000-01-01', periods=total_len, freq=self.freq)
        df_stamp = pd.DataFrame({'date': date_range})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(date_range, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = None

        # Slice the data. For "train" we may use full data; for test/pred, use the later portion.
        if self.train_only or self.flag == 'train':
            border1 = 0
            border2 = total_len
        else:
            # For test/pred, assume last 20% of the time series (with an extra offset for the sequence).
            border1 = int(0.8 * total_len) - self.seq_len
            border2 = total_len

        self.data = data_scaled[border1:border2]
        self.data_stamp = data_stamp[border1:border2] if data_stamp is not None else None

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] if self.data_stamp is not None else None
        seq_y_mark = self.data_stamp[r_begin:r_end] if self.data_stamp is not None else None

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data

class PDE_KS(Dataset):
    def __init__(self, root_path, flag='train', size=None, pair_id=1,
                 features='M', scale=True, timeenc=0, freq=None, train_only=False):
        """
        Dataset class for the PDE_KS spatio-temporal dynamical system.
        
        Parameters:
            root_path: Parent directory containing the PDE_KS folder
            flag: 'train', 'test', or 'pred'. (For simplicity, 'val' is omitted.)
            size: List or tuple [seq_len, label_len, pred_len]. Defaults are used if None.
            pair_id: The experiment pair id to load from the YAML file.
            features: Type indicator, e.g. 'M' for multivariate.
            scale: Whether to scale the data (using StandardScaler).
            timeenc: Either 0 (basic time features) or 1 (advanced encoding).
            freq: Frequency string for time features. If None, delta_t from YAML is used.
            train_only: If True, do not split the data.
        """
        # Set default sequence lengths if not provided.
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.flag = flag
        self.scale = scale
        self.timeenc = timeenc
        self.train_only = train_only
        self.pair_id = pair_id
        self.features = features

        # Define the PDE_KS data folder and YAML configuration file.
        self.data_folder = top_dir.parent.parent.parent / 'data' / 'PDE_KS'
        self.yaml_path = top_dir.parent.parent.parent / 'data' / 'PDE_KS' / 'PDE_KS.yaml'

        self.__load_config__()
        self.__read_data__()

    def __load_config__(self):
        with open(self.yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # Use delta_t from metadata; if freq is not provided, create a timedelta frequency.
        self.delta_t = self.config.get('metadata', {}).get('delta_t', 1.0)
        if not self.timeenc:
            # For basic time features, we simply use date components.
            # Here freq is set as a Timedelta using delta_t seconds.
            self.freq = f"{self.delta_t}S" if self.delta_t else "1S"
        else:
            # For advanced encoding, you might compute time features differently.
            self.freq = f"{self.delta_t}S" if self.delta_t else "1S"
        if self.config.get('type', '') != 'spatio-temporal':
            raise ValueError("Incorrect YAML type; expected 'spatio-temporal'.")
        # Locate the chosen pair configuration by matching the pair id.
        self.pair_config = None
        for pair in self.config.get('pairs', []):
            if pair.get('id') == self.pair_id:
                self.pair_config = pair
                break
        if self.pair_config is None:
            raise ValueError(f"Pair id {self.pair_id} not found in PDE_KS YAML config.")

    def __read_data__(self):
        # Determine the subfolder based on flag. For training, use "train"; for test/pred, use "test".
        folder_flag = 'train' if self.flag == 'train' else 'test'
        if self.flag == 'train':
            files = self.pair_config.get('train', [])
            if not isinstance(files, list):
                files = [files]
        else:
            test_file = self.pair_config.get('test', None)
            if test_file is None:
                raise ValueError("Test file not specified in YAML config.")
            files = [test_file]

        data_list = []
        # Load MAT files from the corresponding subfolder.
        for file_name in files:
            file_path = os.path.join(self.data_folder, folder_flag, file_name)
            mat = scipy.io.loadmat(file_path)
            if 'data' in mat:
                raw = mat['data']
            else:
                keys = [k for k in mat.keys() if not k.startswith('__')]
                if not keys:
                    raise ValueError(f"No valid data in MAT file: {file_path}")
                raw = mat[keys[0]]
            raw = raw.astype(np.float32)
            # According to YAML, the MAT file is shaped as [spatial_dimension, time].
            # Transpose to shape (time, spatial_dimension).
            data_list.append(raw.T)
        # If more than one file is listed, concatenate along the time axis.
        data = np.concatenate(data_list, axis=0) if len(data_list) > 1 else data_list[0]

        # Scale the data if required.
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data

        total_len = data_scaled.shape[0]
        # Create time features.
        # Build a date range using the frequency computed from delta_t.
        date_range = pd.date_range(start='2000-01-01', periods=total_len, freq=self.freq)
        df_stamp = pd.DataFrame({'date': date_range})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(date_range, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = None

        # Define borders for slicing the time series.
        if self.train_only or self.flag == 'train':
            border1 = 0
            border2 = total_len
        else:
            border1 = int(0.8 * total_len) - self.seq_len
            border2 = total_len

        self.data = data_scaled[border1:border2]
        self.data_stamp = data_stamp[border1:border2] if data_stamp is not None else None

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = (self.data_stamp[s_begin:s_end]
                      if self.data_stamp is not None else None)
        seq_y_mark = (self.data_stamp[r_begin:r_end]
                      if self.data_stamp is not None else None)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
