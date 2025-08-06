import os
import torch
import time
import json
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class Experiment(object):
    def __init__(self, args, device,model, optimizer, scheduler, data):
        self.args = args
        self.device = device
        self.epoch = args.epochs
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.seq_length = args.seq_len
        self.pred_length = args.pred_len
        self.label_length = args.label_len
        self.ratio = args.all_ratio
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_path = args.best_path
        self.data = data
        self.weight = torch.from_numpy(np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)).to(self.device)
    def loss_nino(self, y_pred, y_true):
        if len(y_true.shape) > 2:
            y_true = y_true[:, :, 0]
        else:
            y_true = y_true
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()
    def score(self, y_pred, y_true, acc_weight):
        if len(y_true.shape) > 2:
            y_true = y_true[:, :, 0]
        else:
            y_true = y_true
        pred = y_pred - y_pred.mean(dim=0, keepdim=True)
        true = y_true - y_true.mean(dim=0, keepdim=True)
        cor = (pred * true).sum(dim=0) / (torch.sqrt(torch.sum(pred**2, dim=0) * torch.sum(true**2, dim=0)) + 1e-6)
        acc = (acc_weight * cor).sum()
        rmse= torch.mean((y_pred - y_true)**2, dim=0).sqrt()
        sc = 2/3. * acc - rmse.sum()
        return sc.item(), cor, rmse
    def test_score(self, y_pred, y_true, acc_weight):
        with torch.no_grad():
            if len(y_true.shape) > 2:
                y_true = y_true[:, :, 0]
            else:
                y_true = y_true
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)
            true = y_true - y_true.mean(dim=0, keepdim=True)
            cor = (pred * true).sum(dim=0) / (
                        torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0)) + 1e-6)
            acc = (acc_weight * cor).sum()
            rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt()
            sc = 2 / 3. * acc - rmse.sum()
        return sc.item(), cor, rmse
    def one_month(self, y_pred, y_true, acc_weight):
        with torch.no_grad():
            if len(y_true.shape) > 2:
                y_true = y_true[:, :, 0]
            else:
                y_true = y_true
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)
            true = y_true - y_true.mean(dim=0, keepdim=True)
            p = []
            t = []
            length = [1,3,6,9,12,15,18,21]
            for i in length:
                pred_i = pred[:, i-1:i]
                true_i = true[:, i-1:i]
                p.append(pred_i.squeeze(-1))
                t.append(true_i.squeeze(-1))
        return p, t

    def test0(self, setting):
        self.model.load_state_dict(torch.load(os.path.join('/home/kjzhang/0work/AMHF-former/best_model/', 'checkpoint.pth')))
        self.model.eval()
        pred = []
        nino = []
        with torch.no_grad():
            for (batch_idx, (data, label)) in enumerate(self.data):
                input, label = torch.tensor(data).to(self.device).float(), torch.tensor(label).to(self.device).float()
                out, _ = self.model(input, label)
                labels = label[:, -self.pred_length:]
                pred.append(out)
                nino.append(labels)
            nino_pred = torch.cat(pred, dim=0)
            nino_true = torch.cat(nino, dim=0)
            sc, cor, rmse = self.test_score(nino_pred, nino_true, self.weight)
            # pred, true = self.one_month(nino_pred, nino_true, self.weight)
            print('Test score: {:.4f}'.format( sc))
            print('Test rmse:', rmse)
            print('Test correct:', cor)
            folder_path = './results/'  + setting+ '/'

            save_file = folder_path + 'result.txt'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            f = open(save_file, 'a')
            f.write("Custom Parameters: \n")
            f.write(json.dumps(vars(self.args), indent=4))  # 将命令行参数转换为字典并格式化输出
            f.write('\n')
            # f.write(setting + "  \n")
            f.write('score:{}, cor:{}, rmse:{}'.format(sc, cor, rmse))
            # for i in range(8):
            #     f.write('month:{}, pred:{}, true:{}'.format(i, pred[i], true[i]))
            # f.write('pred:{}, true:{}'.format(pred, true))
            # f.write('one_cor:{}'.format(one_month_cor))
            f.write('\n')
            f.write('prediction:{}, observation:{}'.format(nino_pred[0], nino_true[0][:,0]))
            f.write('\n')
            f.close()




