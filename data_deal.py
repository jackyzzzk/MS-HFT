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


def get_feature(data_name, dataset, label, f_num, out_dir=None):
    # select longitudes
    # if data_name == 'GODAS':
    #     data = data[:, :, :, 18:66]
    # else:
    # if data_name == 'GODAS':
    #     n_years = 104
    #     data = dataset
    #     cmipsst = data_transform(data[...,0:1].squeeze(-1),
    #                              n_years)  # train_cmip.sst.values[:]=(2919,36,24,48)  (1692,24,48,21)
    #     # cmipsst = np.expand_dims(cmipsst, axis=-1)
    #     if f_num == 1:
    #         cmipsst = np.expand_dims(cmipsst, axis=-1)
    #     else:
    #         cmiphc = data_transform(data[...,1:2].squeeze(-1), n_years)
    #         cmipsst = np.stack([cmipsst, cmiphc], axis=-1)  # (1692,24,48,21,3)
    #
    #     cmipnino = data_transform(label, n_years)  # (1692,21)
    # else:
    lon = dataset.lon.values
    lon = lon[np.logical_and(lon>=95, lon<=330)]

    data = dataset.sel(lon=lon)
    n_years = 0

    if data_name == "cmip6":
        n_years = 151
    elif data_name == "cmip5":
        n_years = 140
    elif data_name == "SODA":
        n_years = 100
    elif data_name == 'GODAS':
        n_years = 34

    cmipsst = data_transform(data.sst.values[:], n_years)  # train_cmip.sst.values[:]=(2919,36,24,48)  (1692,24,48,21)
    # cmipsst = np.expand_dims(cmipsst, axis=-1)
    if f_num == 1:
        cmipsst = np.expand_dims(cmipsst, axis=-1)
    else:
        cmiphc = data_transform(data.t300.values[:], n_years)

        cmipsst = np.stack([cmipsst, cmiphc], axis=-1)  # (1692,24,48,21,3)


    cmipnino = data_transform(label.nino.values[:], n_years)  # (1692,21)
    # cmip6mon = data_transform(label_cmip.mon.values[:], n_years)
    # cmip6season = data_transform(label_cmip.season.values[:], n_years)
    #
    # cmip6nino = np.stack([cmip6nino, cmip6mon, cmip6season], axis=-1)
    ## (1692,21)
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

    # data.close()
    # label.close()
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
    def __init__(self, sst_cmip6, nino_cmip6, samples_gap):
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

def cmip6_read(train_file, train_label_file, val_file, val_label_file, test_file, test_label_file):
    # test = np.load(test_file)
    # test_label = np.load(test_label_file)
    test = xa.open_dataset(test_file)
    test_label = xa.open_dataset(test_label_file)
    train = xa.open_dataset(train_file)
    train_label = xa.open_dataset(train_label_file)
    val = xa.open_dataset(val_file)
    val_label = xa.open_dataset(val_label_file)
    train_cmip6 = train.where(train.year <= 2265, drop=True)
    train_cmip5 = train.where(train.year > 2265, drop=True)
    train_label_cmip6 = train_label.where(train_label.year <= 2265, drop=True)
    train_label_cmip5 = train_label.where(train_label.year > 2265, drop=True)

    return (train_cmip6, train_label_cmip6), (train_cmip5, train_label_cmip5), (val, val_label), (test, test_label)
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
def del_nan(data):
    data_nan = np.isnan(data)
    data[data_nan] = 0
    print('Number of null after fillna:', np.sum(np.isnan(data)))
    return data
def get_data(cmip6_data_dir, use_data, f_num):
    #cmip6_data_dir = '/data1/kjzhang/data/ENSO_Dataset/cmip6/tianchi_data/'
    cmip6_train_file = os.path.join(cmip6_data_dir, 'enso_round1_train_20210201/CMIP_train.nc')
    cmip6_train_label_file = os.path.join(cmip6_data_dir, 'enso_round1_train_20210201/CMIP_label.nc')
    cmip6_val_file = os.path.join(cmip6_data_dir, 'enso_round1_train_20210201/SODA_train.nc')
    cmip6_val_label_file = os.path.join(cmip6_data_dir, 'enso_round1_train_20210201/SODA_label.nc')
    cmip6_test_file = os.path.join(cmip6_data_dir, 'GODAS_train.nc')
    cmip6_test_label_file = os.path.join(cmip6_data_dir, 'GODAS_label.nc')

    train_6, train_5, val, test = cmip6_read(cmip6_train_file, cmip6_train_label_file, cmip6_val_file, cmip6_val_label_file,
                                                                      cmip6_test_file, cmip6_test_label_file)
    if use_data == 'cmip6':
        train = train_6[0]
        train_label = train_6[1]
        del train_5
    elif use_data == 'cmip5':
        train = train_5[0]
        train_label = train_5[1]
        del train_6
    else:
        raise ValueError(f"Invalid data: {use_data}. Expected 'cmip6', 'cmip5'.")

    feature_train, nino_train = get_feature(use_data, train, train_label, f_num)
    feature_val, nino_val = get_feature('SODA', val[0], val[1], f_num)
    feature_test, nino_test = get_feature('GODAS', test[0], test[1], f_num)
    feature_train = month_and_season(feature_train)
    feature_val = month_and_season(feature_val)
    feature_test = month_and_season(feature_test)
    feature_val = signal(feature_val)
    feature_train = signal(feature_train)
    feature_test = signal(feature_test)
    nino_train = month_and_season(nino_train[..., None, None, None]).squeeze()
    nino_val = month_and_season(nino_val[..., None, None, None]).squeeze(-2).squeeze(-3)
    nino_test = month_and_season(nino_test[..., None, None, None]).squeeze(-2).squeeze(-3)
    feature_train = del_nan(feature_train)
    nino_train = del_nan(nino_train)
    feature_test = del_nan(feature_test)
    nino_test = del_nan(nino_test)
    feature_test = del_nan(feature_test)
    nino_test  = del_nan(nino_test)
    del train, train_label, val, test
    return (feature_train, nino_train), (feature_val, nino_val), (feature_test, nino_test)

def enso_data(data_path, cmip, feature, gap, batch_size, num_workers):
    train, val, test = get_data(data_path, cmip, feature)
    dataset_train = cmip_dataset(train[0], train[1], gap)
    dataset_val = cmip_dataset(val[0], val[1], gap)
    dataset_test = cmip_dataset(test[0], test[1], gap)
    trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valloader = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    testloader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    del train, val
    return trainloader, valloader, testloader

if __name__ == '__main__':
    cmip = 'cmip5'
    mode = 'sst_t300'
    batch_size = 64
    num_workers = 6
    data_path = '/data1/kjzhang/data/ENSO_Dataset/cmip6/tianchi_data/'
    a,b,c = enso_data(data_path,cmip,2,5,2,8)
