from data.data_loader import *
from exp.exp_basic import Exp_Basic
from SLNet.Model import Model
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
import warnings
from loguru import logger
from thop import profile, clever_format

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model = Model(
            self.args.pred_len[-1],
            self.args.layer_num,
            self.args.basic_input,
            self.args.patch_size,
            self.args.Boundary,
            self.args.bins,
            self.args.d_model,
            self.args.dropout,
        ).float()
        input_x = (torch.randn(1, self.args.Boundary[-1] * 2, int(5 * math.log2(self.args.enc_in))).
                   to(self.device))
        GT_x = (torch.randn(1, self.args.pred_len[-1], int(5 * math.log2(self.args.enc_in))).
                to(self.device))
        flops, params = profile(model.to(self.device), inputs=(input_x, GT_x))
        flops, params = clever_format([flops, params], '%.3f')
        logger.info(f"flops：{flops}, params：{params}")

        del model
        model = Model(
            self.args.pred_len[-1],
            self.args.layer_num,
            self.args.basic_input,
            self.args.patch_size,
            self.args.Boundary,
            self.args.bins,
            self.args.d_model,
            self.args.dropout,
        ).float()

        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'Weather': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'Traffic': Dataset_Custom,
        }
        Data = data_dict[self.args.data]

        if flag == 'train':
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = False
            drop_last = True
            batch_size = 1

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            basic_input=args.basic_input,
            pred_len=args.pred_len[-1],
            bins=args.bins,
            missing_ratio=args.missing_ratio
        )
        logger.info(flag)
        logger.info(len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data=None, vali_loader=None):
        self.model.eval()
        total_loss = []
        vali_steps = len(vali_loader)
        logger.info('Validation ...')
        with torch.no_grad():
            iter_count = 0
            time_now = time.time()
            for i, batch_x in enumerate(vali_loader):
                pred, true, _, _ = self._process_one_batch(batch_x, flag='test')
                loss = torch.mean((pred - true) ** 2).detach().cpu().numpy()
                total_loss.append(loss)
                iter_count += 1
                if (i + 1) % 100 == 0:
                    logger.info("\titers: {0} | loss: {1:.4f}".format(i + 1, loss))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (vali_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        train_steps = len(train_loader)

        lr = self.args.learning_rate
        criterion1 = nn.MSELoss().cuda()
        criterion2 = nn.L1Loss().cuda()
        criterion3 = nn.CrossEntropyLoss().cuda()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()

            for i, batch_x in enumerate(train_loader):
                if batch_x.shape[-1] > 100:
                    index_list = np.arange(batch_x.shape[-1])
                    index_list = np.random.choice(index_list,
                                                  int(5 * np.log2(batch_x.shape[-1])), replace=False)
                    c_batch_x = batch_x[:, :, index_list]
                else:
                    c_batch_x = batch_x
                model_optim.zero_grad()
                iter_count += 1
                pred, true, match_score, gt = self._process_one_batch(c_batch_x, flag='train')
                loss = (criterion1(pred, true) + criterion2(pred, true) +
                        0.1 * criterion3(match_score, gt))
                loss.backward(loss)
                model_optim.step()

                if (i + 1) % 100 == 0:
                    logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                                  torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)

            logger.info("Pred_len: {0}| Epoch: {1}, Steps: {2} | Total: Vali Loss: {3:.7f} Test Loss: {4:.7f}| "
                        .format(self.args.pred_len, epoch + 1, train_steps, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
            adjust_learning_rate(model_optim, (epoch + 1), self.args)

        self.args.learning_rate = lr

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=True):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        folder_path = './results/' + setting + f'_ms{self.args.missing_ratio}' + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, batch_x in enumerate(test_loader):
                pred, true, best_value, _ = self._process_one_batch(batch_x, flag='test')

                pred = pred.squeeze(0).detach().cpu().numpy()
                true = true.squeeze(0).detach().cpu().numpy()
                best_value = best_value.squeeze(0).detach().cpu().numpy()
                np.save(folder_path + 'pred_{}.npy'.format(i), pred)
                np.save(folder_path + 'true_{}.npy'.format(i), true)
                np.save(folder_path + 'best_value_{}.npy'.format(i), best_value)

        logger.info("inference time: {}".format(time.time() - time_now))

    def _process_one_batch(self, batch_x, flag):
        batch_x = batch_x.float().to(self.device)
        input_seq = batch_x[:, :-self.args.pred_len[-1], :]
        batch_y = batch_x[:, -self.args.pred_len[-1]:, :]
        if flag == 'train':
            pred_data, match_score, gt = self.model(input_seq, batch_y, flag)
            return pred_data, batch_y, match_score, gt
        else:
            pred_data, best_value, best_index = self.model(input_seq, batch_y, flag)
            return pred_data, batch_y, best_value, best_index
