import os
import torch
import time
import json
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
from logging.handlers import RotatingFileHandler
# handler = RotatingFileHandler(
#     'train.log',
#     mode='w',  # 首次覆盖
#     maxBytes=10*1024*1024,  # 单个文件最大10MB
#     backupCount=5  # 保留5个备份
# )
# logging.basicConfig(handlers=[handler],  level=logging.INFO)
logging.basicConfig(
    filename='train.log',
    filemode='w',  # 关键参数：'w'覆盖，'a'追加（默认）
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


class Trainer(object):
    def __init__(self, args, device,model, optimizer, scheduler, train_data, valid_data, test_data,):
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
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
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
    def vali(self):
        self.model.eval()
        pred = []
        nino = []
        with torch.no_grad():
            for (batch_idx, (data, label)) in enumerate(self.valid_data):
                data, label = data.to(self.device),  label.to(self.device)
                out, _ = self.model(data, label)
                labels = label[:, -self.pred_length:]
                pred.append(out)
                nino.append(labels)
            nino_pred = torch.cat(pred, dim=0)
            nino_true = torch.cat(nino, dim=0)
            sc, cor, rmse = self.score(nino_pred, nino_true, self.weight)
            val_loss = self.loss_nino(nino_pred, nino_true).item()
        return val_loss, sc, cor, rmse.sum()
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

    def test0(self, setting):
        self.model.load_state_dict(torch.load(os.path.join(self.best_path, 'checkpoint.pth')))
        self.model.eval()
        pred = []
        nino = []
        with torch.no_grad():
            for (batch_idx, (data, label)) in enumerate(self.test_data):
                input, label = torch.tensor(data).to(self.device).float(), torch.tensor(label).to(self.device).float()
                out, _ = self.model(input, label)
                labels = label[:, -self.pred_length:]
                pred.append(out)
                nino.append(labels)
            nino_pred = torch.cat(pred, dim=0)
            nino_true = torch.cat(nino, dim=0)
            sc, cor, rmse = self.test_score(nino_pred, nino_true, self.weight)
            print('Test score: {:.4f}'.format( sc))
            print('Test rmse:', rmse)
            print('Test correct:', cor)
            folder_path = './results/' + setting + '/'

            save_file = folder_path + 'result.txt'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            f = open(save_file, 'a')
            f.write("Custom Parameters: \n")
            f.write(json.dumps(vars(self.args), indent=4))  # 将命令行参数转换为字典并格式化输出
            f.write('\n')
            # f.write(setting + "  \n")
            f.write('score:{}, cor:{}, rmse:{}'.format(sc, cor, rmse))
            f.write('\n')
            f.write('\n')
            f.close()

    def train(self):
        # torch.autograd.set_detect_anomaly(True)  # 精准定位问题操作
        start_time = time.time()
        best_score = float('-inf')
        not_improved_count = 0
        best_model_path = os.path.join(self.best_path, 'checkpoint.pth')
        os.makedirs(self.best_path, exist_ok=True)
        for epoch in range(self.epoch):
            torch.cuda.empty_cache()
            iter_count = 0
            train_loss = []
            pred = []
            nino = []
            self.model.train()
            epoch_time = time.time()
            for (batch_idx, (data, label)) in enumerate(tqdm(self.train_data, desc='Epoch:{}'.format(epoch+1), ncols=100, disable=False)):
                iter_count += 1
                inputs, label = data.to(self.device).float(), label.to(self.device).float()
                self.optimizer.zero_grad()
                out, s_loss = self.model(inputs, label,  True)
                labels = label[:, -self.pred_length:]
                if s_loss != 0:
                    loss_nino = self.loss_nino(out, labels)*self.ratio + (1-self.ratio)*s_loss
                else:
                    loss_nino = self.loss_nino(out, labels)
                loss_nino.backward()
                self.optimizer.step()
                if torch.isnan(inputs).any():
                    print(inputs)
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"Invalid value in {name}")
                train_loss.append(loss_nino.item())
                pred.append(out)
                nino.append(labels)

            nino_pred = torch.cat(pred, dim=0)
            nino_true = torch.cat(nino, dim=0)
            epoch_loss = self.loss_nino(nino_pred, nino_true)
            sc, cor, rmse = self.score(nino_pred, nino_true, self.weight)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.4f}')
            print('train score', sc)
            print('batch training cor', cor)


            val_loss, val_sc, val_cor, val_rmse = self.vali()
            print('Validation loss: {:.6f}'.format(val_loss))
            print('Validation score: {:.6f}'.format(val_sc))
            print('Validation cor', val_cor)
            logging.info(
                f"Epoch: {epoch + 1} | "
                f"Cost time: {time.time() - epoch_time:.2f}s | "
                f"Avg Loss: {epoch_loss:.4f} | "
                f"Train Score: {sc} | "
                f"Train Cor: {cor} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Score: {val_sc:.6f} | "
                f"Val Cor: {val_cor}"
            )
            if val_sc > best_score:
                best_score = val_sc
                not_improved_count = 0
                torch.save(self.model.state_dict(), best_model_path)
                print("Saving current best model to " + best_model_path)
                print('Validation rmse', rmse.sum())
                logging.info(f'If best: True|'
                             f'Best  score: {best_score:.6f}')
            else:
                not_improved_count += 1
            self.scheduler.step()


        trainging_time = time.time() - start_time
        print('Total training time: {:.4f}min, best score: {:.6f}'.format((trainging_time / 60), best_score))


