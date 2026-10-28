import os
import re
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_cross_session
from SincAdaptNet import SincAdaptNet


def EA(x):
    from scipy.linalg import fractional_matrix_power
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def train_and_save(data_name, time_length=4, epochs=100, kfolds=10):
    cross_session = False  # or "cross_session"
    # dataset params
    params = {
        'BNCI2014001': {'N': 9, 'chn': 22, 'fs': 250, 'time_lengths': [4]},
        'BNCI2014002': {'N': 14, 'chn': 15, 'fs': 512, 'time_lengths': [5]},
        'BNCI2015001': {'N': 12, 'chn': 13, 'fs': 512, 'time_lengths': [5]}
    }[data_name]

    suffix = "_cross_session" if cross_session else ""
    output_dir = f"baseline_SincAdaptNet_{data_name}{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    results = np.zeros((params['N'], kfolds), dtype=float)
    for subj in range(params['N']):
        # read data
        args = type('A', (), {'data': data_name, 'idt': subj, 'sample_rate': params['fs']})
        if cross_session:
            train_x, train_y, test_x, test_y = read_mi_combine_tar_cross_session(args)
        else:
            train_x, train_y, test_x, test_y = read_mi_combine_tar(args)
        train_x, test_x = EA(train_x), EA(test_x)
        train_x, test_x = train_x[:, None], test_x[:, None]

        for kf in range(kfolds):
            # build dataloaders
            tr_ds = TensorDataset(torch.from_numpy(train_x.astype('float32')),
                                  torch.from_numpy(train_y.squeeze()).long())
            te_ds = TensorDataset(torch.from_numpy(test_x.astype('float32')), torch.from_numpy(test_y).long())
            tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
            te_dl = DataLoader(te_ds, batch_size=32, shuffle=False)

            # init model
            model = SincAdaptNet(params['chn'], 2, 16, 24, 51 if params['fs'] == 250 else 102, fs=params['fs']).cuda()
            criterion = nn.CrossEntropyLoss()
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

            best_acc = 0
            best_wts = None
            for ep in range(epochs):
                model.train()
                for x, y in tr_dl:
                    x, y = x.cuda(), y.cuda()
                    # random crop lengths
                    L = params['fs'] * time_length
                    len = ep % params['time_lengths'][0] + 1
                    x_crop = x[:, :, :, :len * params['fs']]
                    out = model(x_crop)
                    out = out.permute(0, 2, 1).reshape(-1, 2)
                    y_rep = y.repeat_interleave(L)
                    loss = criterion(out, y_rep)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                # eval
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in te_dl:
                        x, y = x.cuda(), y.cuda()
                        out = model(x[:, :, :, :L])
                        out = out.mean(-1)
                        preds = out.argmax(1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)
                acc = 100 * correct / total
                if acc > best_acc:
                    best_acc = acc
                    best_wts = copy.deepcopy(model.state_dict())
            # save
            subj_dir = os.path.join(output_dir, f"sub{subj}", f"{time_length}s")
            os.makedirs(subj_dir, exist_ok=True)
            torch.save(best_wts, os.path.join(subj_dir, f"kf{kf}_model.pkl"))
            results[subj, kf] = best_acc
    # write excel
    df = pd.DataFrame(results, columns=[f'Fold{i + 1}' for i in range(kfolds)])
    df.insert(0, 'Time(s)', f"{time_length}s")
    df.insert(0, 'Subject', range(1, params['N'] + 1))
    df.to_excel(f"{output_dir}.xlsx", index=False)


if __name__ == '__main__':
    # Set torch threads
    cpu_num = os.cpu_count() or 16
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    for ds in ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']:
        train_and_save(ds)
