import argparse
import os
import torch
import numpy as np
import time
from tqdm import tqdm
import shutil
from utils.metrics import metric, KStest
from exp.exp_model import Exp_Model
from loguru import logger

logger.add(
    './info.log',
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    level="INFO",
)

parser = argparse.ArgumentParser(description='Super long time series forecasting network')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--basic_input', type=int, default=336, help='basic input length')
parser.add_argument('--pred_len', type=str, default='96,192,336,720', help='prediction length')

parser.add_argument('--enc_in', type=int, default=7, help='input variable number')
parser.add_argument('--dec_out', type=int, default=7, help='output variable number')
parser.add_argument('--d_model', type=int, default=32, help='hidden dims of model')
parser.add_argument('--layer_num', type=int, default=3, help='model stage number')
parser.add_argument('--patch_size', type=str, default='6,24,168',
                    help='patch size')
parser.add_argument('--Boundary', type=str, default='48,336',
                    help='Boundary for different patch size')

parser.add_argument('--missing_ratio', type=float, default=0.06, help='missing ratio')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--decay', type=float, default=0.5, help='decay rate of learning rate per epoch')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--bins', type=int, default=2000, help='bin num')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--save_loss', action='store_true', help='whether saving results and checkpoints', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_gpu', action='store_true',
                    help='whether to use gpu, it is automatically set to true if gpu is available in your device'
                    , default=False)
parser.add_argument('--train', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

args.pred_len = [int(predl) for predl in args.pred_len.replace(' ', '').split(',')]
args.patch_size = [int(patch) for patch in args.patch_size.replace(' ', '').split(',')]
args.Boundary = [int(boundary) for boundary in args.Boundary.replace(' ', '').split(',')]

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

data_parser = {
    'ECL': {'data': 'ECL.csv', 'root_path': './data/ECL/', 'M': [321, 321]},
    'Traffic': {'data': 'Traffic.csv', 'root_path': './data/Traffic/', 'M': [862, 862]},
    'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'M': [21, 21]},
    'Solar': {'data': 'Solar.csv', 'root_path': './data/Solar/', 'M': [137, 137]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.root_path = data_info['root_path']
    args.enc_in, args.dec_out = data_info['M']

lr = args.learning_rate
logger.info('Args in experiment:')
logger.info(args)

Exp = Exp_Model
for ii in range(args.itr):
    if args.train:
        setting = '{}_pl{}_{}'.format(args.data, args.pred_len, ii)
        logger.info('>>>>>>>start training| pred_len:{}, settings: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.
                    format(args.pred_len, setting))
        try:
            exp = Exp(args)  # set experiments
            exp.train(setting)
        except KeyboardInterrupt:
            logger.info('-' * 99)
            logger.info('Exiting from forecasting early')

        logger.info('>>>>>>>testing| pred_len:{}: {}<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments
        exp.test(setting, load=True)
        torch.cuda.empty_cache()
        args.learning_rate = lr
    else:
        setting = '{}_pl{}_{}'.format(args.data,
                                      args.pred_len, ii)
        logger.info('>>>>>>>testing| pred_len:{} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments

        # exp.test(setting, load=True)
        torch.cuda.empty_cache()
        args.learning_rate = lr

path1 = './result.csv'
if not os.path.exists(path1):
    with open(path1, "a") as f:
        write_csv = ['Time', 'Model', 'Data', 'Pred_len', 'Missing_ratio', 'MSE', 'MAE', 'P_value']
        np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
        f.flush()
        f.close()

logger.info('>>>>>>>writing results<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
first_setting = '{}_pl{}_{}_ms{}'.format(args.data, args.pred_len, 0, args.missing_ratio)
first_folder_path = './results/' + first_setting
num_of_test = len([f for f in os.listdir(first_folder_path)
                   if os.path.isfile(os.path.join(first_folder_path, f)) and 'pred' in f])
logger.info('test windows number: ' + str(num_of_test))

for predl in args.pred_len:
    mses = []
    smapes = []
    p_values = []
    logger.info('Forecasting length = ' + str(predl) + ':')
    for i in tqdm(range(num_of_test)):
        pred_total = 0
        best_value_total = 0
        true = None
        for ii in range(args.itr):
            setting = '{}_pl{}_{}'.format(args.data, args.pred_len, ii)
            folder_path = './results/' + setting + f'_ms{args.missing_ratio}' + '/'
            pred_path = folder_path + 'pred_{}.npy'.format(i)
            best_value_path = folder_path + 'best_value_{}.npy'.format(i)
            pred = np.load(pred_path)
            best_value = np.load(best_value_path)
            pred_total += pred
            best_value_total += best_value
            if true is None:
                true_path = folder_path + 'true_{}.npy'.format(i)
                true = np.load(true_path)
        pred = pred_total / args.itr
        best_value = best_value_total / args.itr
        smape, mse = metric(pred[:predl, :], true[:predl, :])
        p_num = predl // args.patch_size[2]
        p_value = 0
        for k in range(p_num):
            p_value += KStest(np.abs(pred[k * args.patch_size[2]:(k + 1) * args.patch_size[2], :] -
                                     true[k * args.patch_size[2]:(k + 1) * args.patch_size[2], :]),
                              best_value[k * args.patch_size[2]:(k + 1) * args.patch_size[2], :])
        p_value /= p_num
        mses.append(mse)
        smapes.append(smape)
        p_values.append(p_value)

    mse = np.mean(mses)
    smape = np.mean(smapes)
    p_value = np.mean(p_values)
    logger.info('|Mean|mse:{}, smape:{}, p_value:{}'.format(mse, smape, p_value))

    with open(path1, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write(',{},{},{},{},{},{}'.
                format(args.data, predl, args.missing_ratio, mse, smape, p_value) + '\n')
        f.flush()
        f.close()

if not args.save_loss:
    for ii in range(args.itr):
        setting = '{}_pl{}_{}'.format(args.data, args.pred_len, ii)
        dir_path = os.path.join(args.checkpoints, setting)
        check_path = dir_path + '/' + 'checkpoint.pth'
        if os.path.exists(check_path):
            os.remove(check_path)
            os.removedirs(dir_path)

        folder_path = './results/' + setting + f'_ms{args.missing_ratio}'
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
