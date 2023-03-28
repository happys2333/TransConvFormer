import torch
import torch.nn as nn
import model.ours.ours as our
from torch import optim
from util.param import LEARN, BATCH_SIZE, PATIENCE, EPOCH, SEQ_LEN, LABEL_LEN, PRED_LEN, ENCODER_IN, DECODER_IN, \
     OUT_SIZE, OUTPUT_MODEL_PATH,FEATURES,DATASET
from data_process.dataset_process import Process_Dataset
from torch.utils.data import DataLoader
from util.metrics import metric
import os
import time
import numpy as np
from datetime import datetime

DEVICE = torch.device('cuda:0')
LOG_FILE = None


def get_data(flag='train', dataset='ETTh1'):
    process = Process_Dataset(dataset=dataset, seq_len=SEQ_LEN,
                              label_len=LABEL_LEN, pred_len=PRED_LEN, features=FEATURES, target='OT', cols=None, freq='h',
                              timeenc=0, inverse=False, batch_size=BATCH_SIZE,)
    return process.get_data(flag)


def get_loss_fun(LOSS='MSE'):
    criterion = None
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    return criterion


def get_optimizer(model, OPTIMIZER='Adam'):
    optimizer = None
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    return optimizer


def get_model():
    model = our.Ourformer(device=DEVICE, enc_in=ENCODER_IN, dec_in=DECODER_IN, c_out=OUT_SIZE, seq_len=SEQ_LEN,
                              label_len=LABEL_LEN, out_len=PRED_LEN,d_layers=1,e_layers=2,d_ff=2048).to(
        DEVICE)
    return model


def vali(model, vali_data, vali_loader, criterion):
    model.eval()
    total_loss = []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
        pred, true = _process_one_batch(
            model, batch_x, batch_y, batch_x_mark, batch_y_mark)
        loss = criterion(pred.detach().cpu(), true.detach().cpu())
        total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def train(model):
    train_data, train_loader = get_data(flag='train', dataset=DATASET)
    vali_data, vali_loader = get_data(flag='val', dataset=DATASET)
    test_data, test_loader = get_data(flag='test', dataset=DATASET)
    save_path = OUTPUT_MODEL_PATH+"/"+model.name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    time_now = datetime.now().strftime("%Y_%m_%d_%H,%M,%S")
    log_file_name = save_path+"/"+ model.name + "_" + time_now + "_log.txt"
    log_file = open(log_file_name, "w")
    global LOG_FILE
    LOG_FILE= log_file
    model_file_name = save_path+"/"+ model.name + "_" + time_now + ".pt"

    train_steps = len(train_loader)
    print('Model Training Started ...', time_now)
    print('Model Training Started ...', time_now, file=log_file)

    wait = 0
    min_val_loss = np.inf
    loss_func = get_loss_fun()
    print("loss function is MSE")
    print("loss function is MSE",file=log_file)
    opt = get_optimizer(model)
    for epoch in range(EPOCH):
        start_time = datetime.now()
        iter_count = 0
        train_loss = []
        model.train()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            opt.zero_grad()
            pred, true = _process_one_batch(model, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = loss_func(pred, true)
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        end_time = datetime.now()
        print("Epoch: {} cost time: {} s".format(epoch + 1, (end_time - start_time).seconds))
        print("Epoch: {} cost time: {} s".format(epoch + 1, (end_time - start_time).seconds), file=log_file)
        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, loss_func)
        test_loss = vali(model, test_data, test_loader, loss_func)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss), file=log_file)
        if vali_loss < min_val_loss:
            wait = 0
            min_val_loss = vali_loss
            torch.save(model.state_dict(), model_file_name)
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: {}'.format(epoch + 1))
                print('Early stopping at epoch: {}'.format(epoch + 1), file=log_file)
                break

    model.load_state_dict(torch.load(model_file_name))
    print('Model Training Ended ...', time.ctime(), file=log_file)
    print('Model Training Ended ...', time.ctime())
    return model


def test(model):

    test_data, test_loader = get_data(flag='test', dataset=DATASET)
    print("Model Testing Started ...",datetime.now().strftime("%Y_%m_%d_%H,%M,%S"))
    print("Model Testing Started ...", datetime.now().strftime("%Y_%m_%d_%H,%M,%S"),file=LOG_FILE)
    model.eval()

    preds = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        pred, true = _process_one_batch(
            model, batch_x, batch_y, batch_x_mark, batch_y_mark)
        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)
    save_path = OUTPUT_MODEL_PATH+"/"+ model.name
    save_path = save_path+"/"+ "test_result"


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # result save
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae,rmse,mape))
    print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape),file=LOG_FILE)
    np.save(save_path + '/metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(save_path + '/pred.npy', preds)
    np.save(save_path + '/true.npy', trues)
    print("Model Testing Ended ...", datetime.now().strftime("%Y_%m_%d_%H,%M,%S"))
    print("Model Testing Ended ...", datetime.now().strftime("%Y_%m_%d_%H,%M,%S"),file=LOG_FILE)
    return


def predict(model):
    pred_data, pred_loader = get_data(flag='pred')
    model.eval()
    preds = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
        pred, true = _process_one_batch(
            pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
        preds.append(pred.detach().cpu().numpy())

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    save_path = OUTPUT_MODEL_PATH+"/"+ model.name
    save_path = save_path+"/"+ "pred_result"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # result save
    np.save(save_path + '/real_prediction.npy', preds)

    return


def _process_one_batch(model, batch_x, batch_y, batch_x_mark, batch_y_mark):
    batch_x = batch_x.float().to(DEVICE)
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float().to(DEVICE)
    batch_y_mark = batch_y_mark.float().to(DEVICE)

    dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:, :LABEL_LEN, :], dec_inp], dim=1).float().to(DEVICE)

    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    batch_y = batch_y[:, -PRED_LEN:, 0:].to(DEVICE)

    return outputs, batch_y


if __name__ == '__main__':
    our = get_model()
    best_model = train(our)
    test(best_model)

    torch.cuda.empty_cache()

