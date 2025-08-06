import os
import numpy as np
import xarray as xa
import torch
import numpy
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import argparse
def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    # input_gap=1: time gaps between two consecutive input frames
    # input_length=12: the number of input frames
    # pred_shift=26: the lead_time of the last target to be predicted
    # pred_length=26: the number of frames to be predicted
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span+pred_shift-1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length+pred_length), dtype=int)
    return ind[::samples_gap]


def fold(data, size=36, stride=12):
    # inverse of unfold/sliding window operation
    # only applicable to the case where the size of the sliding windows is n*stride
    # data (N, size, *)
    # outdata (N_, *)
    # N/size is the number/width of sliding blocks
    assert size % stride == 0
    times = size // stride
    remain = (data.shape[0] - 1) % times
    if remain > 0:
        ls = list(data[::times]) + [data[-1, -(remain*stride):]]
        outdata = np.concatenate(ls, axis=0)  # (36*(151//3+1)+remain*stride, *, 15)
    else:
        outdata = np.concatenate(data[::times], axis=0)  # (36*(151/3+1), *, 15)
    assert outdata.shape[0] == size * ((data.shape[0]-1)//times+1) + remain * stride
    return outdata


def data_transform(data, num_years_per_model):
    # data (2919, 36, 24, 72)
    # num_years_per_model: 139
    length = data.shape[0]
    assert length % num_years_per_model == 0
    num_models = length // num_years_per_model
    outdata = np.stack(np.split(data, length/num_years_per_model, axis=0), axis=-1)  # (151, 36, 24, 48, 15)
    outdata = fold(outdata, data.shape[1], 12)  # (1692,24,48,21)  # 起到的作用实际上就是将138年12个月的数据和第139年36个月的数据拼接在一起，得到1692个月的数据

    # check output data
    assert outdata.shape[-1] == num_models
    # assert not np.any(np.isnan(outdata))
    return outdata


def get_feature(dataset, label, f_num, n_years = 34, out_dir=None):
    lon = dataset.lon.values
    lon = lon[np.logical_and(lon>=95, lon<=330)]
    data = dataset.sel(lon=lon)


    cmipsst = data_transform(data.sst.values[:], n_years)  # train_cmip.sst.values[:]=(2919,36,24,48)  (1692,24,48,21)
    # cmipsst = np.expand_dims(cmipsst, axis=-1)
    if f_num == 1:
        cmipsst = np.expand_dims(cmipsst, axis=-1)
    else:
        cmiphc = data_transform(data.t300.values[:], n_years)

        cmipsst = np.stack([cmipsst, cmiphc], axis=-1)  # (1692,24,48,21,3)
    cmipnino = data_transform(label.nino.values[:], n_years)  # (1692,21)
    assert len(cmipsst.shape) == 5
    assert len(cmipnino.shape) == 2
    # store processed data for faster data access
    if out_dir is not None:
        ds_cmip6 = xa.Dataset({'sst': (['month', 'lat', 'lon', 'model'], cmipsst),
                               'nino': (['month', 'model'], cmipnino)},
                              coords={'month': np.repeat(np.arange(1, 13)[None], cmipnino.shape[0] // 12, axis=0).flatten(),
                                      'lat': data.lat.values, 'lon': data.lon.values,
                                      'model': np.arange(15)+1})
        ds_cmip6.to_netcdf(Path(out_dir) / 'cmip6.nc')
    return cmipsst, cmipnino

def cat_over_last_dim(data):
    ## data=(331,38,24,48,21,3)
    if len(data.shape)>=5:
        return np.concatenate(np.moveaxis(data, 4, 0), axis=0)
    else:
        return np.concatenate(np.moveaxis(data, 2, 0), axis=0)
def time_add(data):
    data = data[:,:,:,:,:1]
    month_max = 12
    season_max = 3
    month_start = -1
    season_start = 0
    month_data = np.zeros_like(data)
    season_data = np.zeros_like(data)
    month_init = month_start
    season_init = season_start
    for index in range(data.shape[0]):
        if (index) % month_max == 0:
            month_init = month_start
        month_init = month_init + 1
        if (index) % season_max == 0:
            season_init = season_init + 1
        if month_init > month_max:
            month_init = 0
        if season_init > season_max:
            season_init = 0
        month_data[index:index + 1,:, :,:] = month_init
        season_data[index:index + 1,:, :,:] = season_init
    return month_data, season_data

class cmip_dataset(Dataset):
    def __init__(self, sst_cmip6, nino_cmip6, samples_gap, start):
        super().__init__()
        # cmip6 (1692,24,48,21,3)  nino_cmip6=(1692,21, 3)
        sst = []
        target_nino = []
        if sst_cmip6 is not None:
            assert len(sst_cmip6.shape) == 5
            assert len(nino_cmip6.shape) == 3
            # assert len(nino_cmip6.shape) == 2
            idx_sst = prepare_inputs_targets(sst_cmip6.shape[0], input_gap=1, input_length=12,
                                             pred_shift=24, pred_length=24, samples_gap=samples_gap)
            # print(sst_cmip6[idx_sst].shape)
            idx_sst = idx_sst[start:]
            sst.append(cat_over_last_dim(sst_cmip6[idx_sst]))  # sst_cmip6[idx_sst]=(331,38,24,48,21,4)
            # target_nino.append(cat_over_last_dim(nino_cmip6[idx_sst[:, 12:36]]))  # nino_cmip6[idx_sst]=(331,38,21,3)
            target_nino.append(cat_over_last_dim(nino_cmip6[idx_sst]))

        # sst data containing both the input and target
        self.sst = np.concatenate(sst, axis=0)  # (N, 38, lat, lon, 3)
        # nino data containing the target only
        self.target_nino = np.concatenate(target_nino, axis=0)  # (N, 24)
        # print(self.sst.shape)
        # print(self.target_nino.shape)
        assert self.sst.shape[0] == self.target_nino.shape[0]
        assert self.sst.shape[1] == 36
        # assert self.target_nino.shape[1] == 24
        assert self.target_nino.shape[1] == 36
        self.sst = torch.tensor(self.sst, dtype=torch.float32)
        self.target_nino = torch.tensor(self.target_nino, dtype=torch.float32)

    def GetDataShape(self):
        return {'sst': self.sst.shape,
                'nino target': self.target_nino.shape}

    def __len__(self):
        return self.sst.shape[0]

    def __getitem__(self, idx):
        return self.sst[idx], self.target_nino[idx]



def pad_collate_fn(batch):
    seq_x, seq_x_t, seq_y = zip(*batch)
    seq_x = torch.nn.utils.rnn.pad_sequence(seq_x, batch_first=True)
    seq_x_t = torch.nn.utils.rnn.pad_sequence(seq_x_t, batch_first=True)
    seq_y = torch.nn.utils.rnn.pad_sequence(seq_y, batch_first=True)
    return seq_x, seq_x_t, seq_y
def read_file(file_path):
    npy_data = []
    folder_path = Path(file_path)
    # 遍历并加载所有 .npy 文件
    for npy_file in folder_path.rglob("*.npy"):  # 递归查找所有 .npy 文件
        data = np.load(npy_file)  # 加载 .npy 文件内容
        npy_data.append(data)
    npy_data = np.stack(npy_data, axis=0)
    return npy_data
def cmip6_read(train_file, train_label_file):
    train = xa.open_dataset(train_file, decode_times=False)
    train_label = xa.open_dataset(train_label_file, decode_times=False)
    lana = train.where((train.year >= 5) & (train.year <= 7), drop=True)
    lana_label = train_label.where((train_label.year >= 5) & (train_label.year <= 7), drop=True)
    el96 = train.where((train.year >= 14) & (train.year <= 16), drop=True)
    el96_label = train_label.where((train_label.year >= 14) & (train_label.year <= 16), drop=True)
    el15 = train.where((train.year >= 31) & (train.year <= 33), drop=True)
    el15_label = train_label.where((train_label.year >= 31) & (train_label.year <= 33), drop=True)
    return (train, train_label), (lana, lana_label), (el96, el96_label), (el15, el15_label)
def signal(data):
    nino = np.zeros_like(data[:, :, :, :, 0:1])
    nino[:, 10:13, 15:20, :, 0:1] = 1
    data_with_label = np.concatenate([
        data,
        nino  # 新增一个维度作为特征
    ], axis=-1)
    return data_with_label
def month_and_season(data):
    month, season = time_add(data)
    return np.concatenate((data, month, season), axis=-1)
def get_data(cmip_data_dir, use_data, f_num):
    godas_file = os.path.join(cmip_data_dir, 'GODAS_train.nc')
    label_file = os.path.join(cmip_data_dir, 'GODAS_label.nc')
    # label_file2 = os.path.join(cmip_data_dir, 'GODAS.label.12mn_3mv.1982_2017.nc')
    # label_file = os.path.join(cmip_data_dir, 'GODAS.label.type.DJF.1982_2017.nc')

    godas, la, el19, el20 = cmip6_read(godas_file, label_file)
    feature_godas, nino_godas = get_feature(godas[0], godas[1], f_num)
    feature_train = month_and_season(feature_godas)
    feature_train = signal(feature_train)
    nino_train = month_and_season(nino_godas[..., None, None, None]).squeeze(axis=(-2, -3))

    feature_la, nino_la = get_feature(la[0], la[1], f_num, 3)
    feature_la = signal(month_and_season(feature_la))
    nino_la = month_and_season(nino_la[..., None, None, None]).squeeze(axis=(-2, -3))
    feature_el19, nino_el19 = get_feature(el19[0], el19[1], f_num, 3)
    feature_el19 = signal(month_and_season(feature_el19))
    nino_el19 = month_and_season(nino_el19[..., None, None, None]).squeeze(axis=(-2, -3))
    feature_el20, nino_el20 = get_feature(el20[0], el20[1], f_num, 3)
    feature_el20 = signal(month_and_season(feature_el20))
    nino_el20 = month_and_season(nino_el20[..., None, None, None]).squeeze(axis=(-2, -3))
    return (feature_train, nino_train), (feature_la, nino_la), (feature_el19, nino_el19), (feature_el20, nino_el20)

def enso_data(data_path, cmip, feature, gap, batch_size, num_workers, start, enso_events):
    train_godas, la, el19, el20 = get_data(data_path, cmip, feature)
    if enso_events == 'la':
        train = (la[0], la[1])
    elif enso_events == 'el19':
        train = (el19[0], el19[1])
    elif enso_events == 'el20':
        train = (el20[0], el20[1])
    else:
        train = (train_godas[0], train_godas[1])
    dataset_train = cmip_dataset(train[0], train[1], gap, start)

    trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return trainloader

if __name__ == '__main__':
    cmip = 'cmip5'
    mode = 'sst_t300'
    batch_size = 64
    num_workers = 6
    # train = xa.open_dataset('F://0work/ENSO_dataset/cmip5/CMIP5_CNN/CMIP5.input.36mn.1861_2001.nc')
    # train_label = xa.open_dataset('F://0work/ENSO_dataset/cmip5/CMIP5_CNN/CMIP5.label.12mn_2mv.1982_2017.nc')
    # train_label2 = xa.open_dataset('F://0work/ENSO_dataset/cmip5/CMIP5_CNN/CMIP5.label.12mn_3mv.1982_2017.nc')
    # val = xa.open_dataset('F://0work/ENSO_dataset/cmip5/SODA/SODA.input.36mn.1871_1970.nc')
    # val_label = xa.open_dataset('F://0work/ENSO_dataset/cmip5/SODA/SODA.label.nino34.12mn_2mv.1873_1972.nc')
    # val_label2 = xa.open_dataset('F://0work/ENSO_dataset/cmip5/SODA/SODA.label.nino34.12mn_3mv.1873_1972.nc')
    data_path = 'F://0work/ENSO_dataset/tianchi_data/'
    a = enso_data(data_path,cmip,2,1,2,0)
