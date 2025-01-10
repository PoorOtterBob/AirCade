import os
import sys
import random
sys.path.append(os.path.abspath(__file__ + '/../../..'))
from src.utils.util import config, file_dir
from src.utils.graph import Graph
from src.utils.dataset import HazeData
from src.utils.args import get_public_config
import arrow
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
import numpy as np
from src.models.AirCade import AirCade
torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
sys.path.remove(os.path.abspath(__file__ + '/../../..'))
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_config():
    parser = get_public_config()
    args = parser.parse_args()
    return args

args = get_config()
graph = Graph()
city_num = graph.node_num
batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = args.num
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
args.node_num = city_num
args.horizon = pred_len
exp_model = 'staeformer'
criterion = nn.MSELoss()

# Default is OOD
train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test' if args.tood else 'Test_default')

in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std

# Get dataset information and node about training and test
node_num=city_num
set_seed(args.seed) 
node_order = np.arange(node_num)
np.random.shuffle(node_order) 
node_num_training = int(node_num / (1.+args.max_increase_ratio))
print('Using', node_num_training, 'nodes in training')
node_num_test_increase = int(node_num_training * args.test_increase_ratio)
node_num_test_decrease = int(node_num_training * args.test_decrease_ratio)
node_num_test_difference = node_num_test_increase
node_training = node_order[:node_num_training]
node_test = np.concatenate([node_order[:node_num_training-node_num_test_decrease], 
                            node_order[node_num_training:node_num_training+node_num_test_increase]])


def masked_mape(preds, labels, null_val):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs((preds - labels) / labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)

def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    null_val = label.min() if label.min() < 1 else np.array(0)
    mape = masked_mape(predict, label, null_val)
    return mae, rmse, mape, csi, pod, far

def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info

def train(train_loader, model, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()      
        pm25, feature, time_arr, day, week = data
        pm25 = pm25[..., node_training, :]    
        feature1 = feature[..., hist_len : , node_training, :]
        feature = feature[..., : hist_len, node_training, :]  
        pm25 = pm25.to(device)
        feature = feature.to(device)
        feature1 = feature1.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(torch.cat([pm25_hist, feature], dim=-1), feature1)
        loss = (torch.fft.rfft(pm25_pred, dim=1) - torch.fft.rfft(pm25_label, dim=1)).abs().mean() 
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx + 1
    return train_loss

def val(val_loader, model, device):
    model.eval()
    val_loss = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
        pm25, feature, time_arr, day, week = data
        pm25 = pm25[..., node_training, :]
        feature1 = feature[..., hist_len:, node_training, :]
        feature = feature[..., :hist_len, node_training, :] 
        pm25 = pm25.to(device)
        feature = feature.to(device)
        feature1 = feature1.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(torch.cat([pm25_hist, feature], dim=-1), feature1)
        loss = (torch.fft.rfft(pm25_pred, dim=1) - torch.fft.rfft(pm25_label, dim=1)).abs().mean() 
        val_loss += loss.item()
    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, model, device):
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr, day, week = data
        pm25 = pm25[..., node_test if args.sood else node_training, :]
        feature1 =  feature[..., hist_len:, node_test if args.sood else node_training, :] 
        feature = feature[..., :hist_len, node_test if args.sood else node_training, :]
        pm25 = pm25.to(device)
        feature = feature.to(device)
        feature1 = feature1.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(torch.cat([pm25_hist, feature], dim=-1), feature1)
        loss = (torch.fft.rfft(pm25_pred, dim=1) - torch.fft.rfft(pm25_label, dim=1)).abs().mean() 
        test_loss += loss.item()
        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())
    test_loss /= batch_idx + 1
    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    #if args.sood:
    predict_epoch1 = predict_epoch[..., :-node_num_test_increase, :] if node_num_test_increase and args.sood else predict_epoch
    label_epoch1 = label_epoch[..., :-node_num_test_increase, :] if node_num_test_increase and args.sood else predict_epoch
    predict_epoch2 = predict_epoch[..., -node_num_test_increase:, :] if node_num_test_increase and args.sood else None
    label_epoch2 = label_epoch[..., -node_num_test_increase:, :] if node_num_test_increase and args.sood else None
    return test_loss, predict_epoch, label_epoch, time_epoch, predict_epoch1, label_epoch1, predict_epoch2, label_epoch2

def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def main():
    exp_info = get_exp_info()
    print(exp_info)
    set_seed(3028)
    device = torch.device(args.device)
    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, mape_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], [], []
    if args.sood:
        rmse_list1, mae_list1, mape_list1, csi_list1, pod_list1, far_list1 = [], [], [], [], [], []
        rmse_list2, mae_list2, mape_list2, csi_list2, pod_list2, far_list2 = [], [], [], [], [], []
    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        
        model = AirCade(input_dim=in_dim, output_dim=args.output_dim)

        model = model.to(device)

        model_name = type(model).__name__

        print(str(model))

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, model, optimizer, device)
            val_loss = val(val_loader, model, device)

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)

            if epoch - best_epoch > early_stop:
                break

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                test_loss, predict_epoch, label_epoch, time_epoch, predict_epoch1, label_epoch1, predict_epoch2, label_epoch2, = test(test_loader, model, device)
                train_loss_, val_loss_ = train_loss, val_loss
                mae, rmse, mape, csi, pod, far = get_metric(predict_epoch, label_epoch)
                print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, MAE: %0.4f, RMSE: %0.4f, MAPE: %0.4f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, val_loss_, test_loss, mae, rmse, mape, csi, pod, far))
                if args.sood:
                    mae1, rmse1, mape1, csi1, pod1, far1 = get_metric(predict_epoch1, label_epoch1)
                    mae2, rmse2, mape2, csi2, pod2, far2 = get_metric(predict_epoch2, label_epoch2)
                if save_npy:
                    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
                    np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)
                    if args.sood:
                        np.save(os.path.join(exp_model_dir, 'predict1.npy'), predict_epoch1)
                        np.save(os.path.join(exp_model_dir, 'label1.npy'), label_epoch1)
                        np.save(os.path.join(exp_model_dir, 'predict2.npy'), predict_epoch2)
                        np.save(os.path.join(exp_model_dir, 'label2.npy'), label_epoch2)


        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)
        if args.sood:
            rmse_list1.append(rmse1)
            mae_list1.append(mae1)
            mape_list1.append(mape1)
            csi_list1.append(csi1)
            pod_list1.append(pod1)
            far_list1.append(far1)

            rmse_list2.append(rmse2)
            mae_list2.append(mae2)
            mape_list2.append(mape2)
            csi_list2.append(csi2)
            pod_list2.append(pod2)
            far_list2.append(far2)

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, MAE: %0.4f, RMSE: %0.4f, MAPE: %0.4f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            train_loss_, val_loss_, test_loss, mae, rmse, mape, csi, pod, far))
        if args.sood:
            print(
                'MAE1: %0.4f, RMSE1: %0.4f, MAPE1: %0.4f, CSI1: %0.4f, POD1: %0.4f, FAR1: %0.4f' % (
                mae1, rmse1, mape1, csi1, pod1, far1))
            print(
                'MAE2: %0.4f, RMSE2: %0.4f, MAPE2: %0.4f, CSI2: %0.4f, POD2: %0.4f, FAR2: %0.4f' % (
                mae2, rmse2, mape2, csi2, pod2, far2))
            
    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAPE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(mape_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list))
    if args.sood:
        exp_metric_str1 = '---------------------------------------\n' + \
                        'MAE1        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list1)) + \
                        'RMSE1       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list1)) + \
                        'MAPE1       | mean: %0.4f std: %0.4f\n' % (get_mean_std(mape_list1)) + \
                        'CSI1        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list1)) + \
                        'POD1        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list1)) + \
                        'FAR1        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list1))
        
        exp_metric_str2 = '---------------------------------------\n' + \
                        'MAE2        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list2)) + \
                        'RMSE2       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list2)) + \
                        'MAPE2       | mean: %0.4f std: %0.4f\n' % (get_mean_std(mape_list2)) + \
                        'CSI2       | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list2)) + \
                        'POD2        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list2)) + \
                        'FAR2        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list2))

    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)
        if args.sood:
            f.write(exp_metric_str1)
            f.write(exp_metric_str2)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    if args.sood:
        print(exp_metric_str1)
        print(exp_metric_str2)
    print(str(model))
    print(metric_fp)
        
if __name__ == '__main__':
    main()