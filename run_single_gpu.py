import torch
import os
from torch import optim
import random
import numpy as np
from trainer import Trainer
from params import parse_args as pa
from data_deal import enso_data
from model import model as model
def set_seed(seed):
    # 固定随机种子等操作
    seed_n = seed
    print('seed is ' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    os.environ['PYTHONHASHSEED'] = str(seed_n)  # 为了禁止hash随机化，使得实验可复现。
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
def data_loader(args):
    path = args.data_path
    cmip = args.cmip
    f_num = args.f_num
    gap = args.gap
    train_loader, val_loader, test_loader = enso_data(path, cmip, f_num, gap,args.batch_size, args.num_workers)
    return train_loader, val_loader, test_loader
if __name__ == '__main__':
    args = pa()
    set_seed(args.seed)
    device, device_ids = get_device(args)
    model = model(args, device)
    train_loader, val_loader, test_loader = data_loader(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.lr_decay_steps,
                                               gamma=args.lr_decay_rate)
    model = model.to(device)
    enso = Trainer(args, device, model, optimizer, schedular, train_loader, val_loader, test_loader)
    print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    enso.train()
    setting = '{}_{}_{}_{}_{}_{}'.format(args.batch_size, args.epochs, args.feature,args.num_caps,
                                                     args.num_route, args.d_model)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # enso.test(setting=setting)
    enso.test0(setting)
