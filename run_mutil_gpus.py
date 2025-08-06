import torch.nn as nn
import torch.multiprocessing as mp
import torch
import os
from torch.utils.data import DataLoader
from torch.distributed import init_process_group
from torch import optim
from trainer import Trainer
from params import parse_args as pa
from data_deal import enso_data
from model import model as stmodel
def get_device(args):
    if torch.cuda.is_available() and args.use_gpu:
        if args.use_multi_gpu:
            #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.multi_gpus)
            device_ids = [int(id_) for id_ in args.device_ids.replace(' ', '').split(',')]
            # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
            args.gpu = device_ids[0]
            print('\n')
            print(f'Using multiple GPUs: {device_ids} with primary GPU: cuda:{args.gpu}')
            return torch.device(f'cuda:{args.gpu}'), device_ids
        else:
            print(f'Using single GPU: cuda:{args.gpu}')
            return torch.device('cuda:{}'.format(args.gpu)), 0
    else:
        print('Using CPU')
        return torch.device('cpu'), 0
def ddp_setup(rank, world_size):
    """
    Args:
        rank: 进程的唯一标识，在 init_process_group 中用于指定当前进程标识
        world_size: 进程总数
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
def get_visible_devices(device_count):
    return [int(d) for d in range(len(device_count))]


def collate_fn(batch):
    # 假设数据是时间序列
    data, target = zip(*batch)

    # 假设数据是列表，长度不一致，找到最大长度
    max_len = max([len(d) for d in data])

    # 填充数据
    padded_data = [torch.cat([d, torch.zeros(max_len - len(d))]) for d in data]

    # 如果需要填充目标数据，也可以类似处理
    padded_target = [torch.cat([t, torch.zeros(max_len - len(t))]) for t in target]

    # 堆叠数据
    data_tensor = torch.stack(padded_data, dim=0)
    target_tensor = torch.stack(padded_target, dim=0)

    return data_tensor, target_tensor
def data_loader(args):
    path = args.data_path
    cmip = args.cmip
    f_num = args.f_num
    gap = args.gap
    train_loader, val_loader, test_loader = enso_data(path, cmip, f_num, gap,args.batch_size, args.num_workers)
    return train_loader, val_loader, test_loader
def main(rank, world_size, device_ids):
    local_device = device_ids[rank]  # 根据 rank 分配 GPU
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 设置当前进程的 GPU
    print(f"[Rank {rank}] Using GPU: {local_device}")
    devices = get_visible_devices(device_ids)
    device = devices[0]

    args = pa()
    # device, device_ids = get_device(args)
    model = stmodel(args, device)
    train_data, val_data, test_data = data_loader(args)
    train_sample = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=(train_sample is None),
                              sampler=train_sample, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_sample = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=(val_sample is None),
                              sampler=val_sample, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_sample = torch.utils.data.distributed.DistributedSampler(test_data)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=(test_sample is None),
                              sampler=test_sample, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.lr_decay_steps,
                                               gamma=args.lr_decay_rate)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=devices, output_device=device)
    enso = Trainer(args, device, model, optimizer, schedular, train_loader, val_loader, test_loader)
    print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    enso.train()
    setting = '{}_{}_{}_{}_{}_{}_{}'.format(args.batch_size, args.epochs, args.feature, args.seq_len, args.pred_len,
                                        args.num_caps, args.dim_caps)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    enso.test(setting=setting)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
    args = pa()
    device, device_ids = get_device(args)
    world_size = len(device_ids)
    mp.spawn(main, args=(world_size, device_ids), nprocs=world_size)
