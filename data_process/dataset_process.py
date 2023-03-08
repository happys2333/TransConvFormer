from data_process.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred

import numpy as np

from torch.utils.data import DataLoader

import warnings
import os

warnings.filterwarnings('ignore')
from util.param import ECL_PATH, WTH_PATH, KDD_PATH, ETT_PATH_DIR


class Process_Dataset:
    def __init__(self, dataset, batch_size,  seq_len, label_len, pred_len, features, target, cols, freq,
                 timeenc, inverse):
        self.dataset = dataset
        self.batch_size = batch_size
        data_path_dict = {
            'ETTh1':  os.path.join(ETT_PATH_DIR,"ETTh1.csv"),
            'ETTh2':  os.path.join(ETT_PATH_DIR,"ETTh2.csv"),
            'ETTm1':  os.path.join(ETT_PATH_DIR,"ETTm1.csv"),
            'ETTm2':  os.path.join(ETT_PATH_DIR,"ETTm2.csv"),
            'WTH': WTH_PATH,
            'ECL': ECL_PATH,
            'KDD': KDD_PATH,
        }
        self.data_path = data_path_dict[dataset]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.cols = cols
        self.freq = freq
        self.timeenc = timeenc
        self.inverse = inverse

    def get_data(self, flag):
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'KDD': Dataset_Custom,
        }
        Data = data_dict[self.dataset]
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = self.batch_size
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.batch_size
        data_set = Data(
            data_path=self.data_path,
            flag=flag,
            size=[self.seq_len, self.label_len, self.pred_len],
            features=self.features,
            target=self.target,
            inverse=self.inverse,
            timeenc=self.timeenc,
            freq=self.freq,
            cols=self.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last)
        return data_set, data_loader


if __name__ == "__main__":
    dataset = Process_Dataset(dataset="ETTh1", seq_len = 24 * 4 * 4, label_len = 24 * 4, pred_len = 24 * 4, features ='S', target='OT', cols=None, freq = 'h',
                 timeenc = 0, inverse = False, batch_size=1)
    print(dataset.dataset)
    print(dataset.get_data("train"))
