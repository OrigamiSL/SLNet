import os
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path='ECL.csv', flag='train',
                 basic_input=336, pred_len=720, bins=2000, missing_ratio=0.06):
        # info
        self.basic_input = basic_input
        self.pred_len = pred_len
        self.bins = bins
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.missing_ratio = missing_ratio
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        if self.data_path == 'Wind.csv':
            df_raw = df_raw[cols]
        else:
            cols.remove('date')
            df_raw = df_raw[['date'] + cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data_x = data[border1:border2]
        self.data_train = data[border1s[0]:border2s[0]]
        self.data_train_val = data[border1s[0]:border2s[1]]

    def __getitem__(self, index):
        seq_x = None
        # Quantitative results
        # if self.set_type == 0:
        #     r_end = index + self.basic_input + self.pred_len
        #     seq_x = self.data_x[:r_end]
        #     seq_x_pad = np.zeros([self.data_train.shape[0] - seq_x.shape[0], seq_x.shape[1]])
        #     seq_x = np.concatenate([seq_x_pad, seq_x], axis=0)
        # elif self.set_type == 1:
        #     r_end = index + self.pred_len
        #     seq_x_0 = self.data_train
        #     seq_x_1 = self.data_x[:r_end]
        #     seq_x = np.concatenate([seq_x_0, seq_x_1], axis=0)
        # elif self.set_type == 2:
        #     r_end = index + self.pred_len
        #     seq_x_0 = self.data_train_val
        #     seq_x_1 = self.data_x[:r_end]
        #     seq_x = np.concatenate([seq_x_0, seq_x_1], axis=0)

        # Efficiency analysis
        if self.set_type == 0:
            r_end = index + self.basic_input + self.pred_len
            if r_end < 2688 + self.pred_len:
                seq_x = self.data_x[:r_end]
                seq_x_pad = np.zeros([2688 + self.pred_len - seq_x.shape[0], seq_x.shape[1]])
                seq_x = np.concatenate([seq_x_pad, seq_x], axis=0)
            else:
                seq_x = self.data_x[r_end - 2688 - self.pred_len:r_end]
        elif self.set_type == 1:
            r_end = index + self.pred_len
            seq_x_0 = self.data_train
            seq_x_1 = self.data_x[:r_end]
            seq_x = np.concatenate([seq_x_0, seq_x_1], axis=0)[- 2688 - self.pred_len:]
        elif self.set_type == 2:
            r_end = index + self.pred_len
            seq_x_0 = self.data_train_val
            seq_x_1 = self.data_x[:r_end]
            seq_x = np.concatenate([seq_x_0, seq_x_1], axis=0)[- 2688 - self.pred_len:]
        if self.missing_ratio > 0:
            current_missing_ratio = self.missing_ratio
            if self.set_type == 0:
                current_missing_ratio = np.random.random() * self.missing_ratio
            L, V = seq_x.shape
            zero_matrix = np.zeros_like(seq_x[:-self.pred_len, :])
            mask_matrix = np.random.random((L - self.pred_len, V))
            mask_x = np.where(mask_matrix < current_missing_ratio,
                              zero_matrix, seq_x[:-self.pred_len, :])
            seq_x = np.concatenate([mask_x, seq_x[-self.pred_len:]], axis=0)

        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return len(self.data_x) - self.basic_input - self.pred_len + 1
        else:
            return len(self.data_x) - self.pred_len + 1
