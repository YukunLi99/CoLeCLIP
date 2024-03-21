import os
import argparse
import numpy as np
import pandas as pd



def main(args):
    order = [
        ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"],
        ['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']
    ]
    order = order[args.Order]
    print(order)
    res = pd.read_csv(args.file_path)
    # Read results in order of tasks
    row_id = []
    for task in order:
        row_id.append(res[res.stage == task].index.tolist()[0])

    res = res.iloc[row_id].iloc[:, 1:].values   # 11 * 11
    assert res.shape[0] == res.shape[1] and res.shape[0] == len(order)


    # mask
    mask = np.zeros_like(res)
    for i in range(len(order)-1):
        mask += np.eye(11, k=i+1)

    # transfer
    transfer = (res * mask)[:-1, 1:].sum(axis=0) / mask[:-1, 1:].sum(axis=0)
    tol_transfer = transfer.mean()

    # average
    avg = res.mean(axis=0)
    tol_avg = avg.mean()

    # last
    last = res[-1]
    tol_last = last.mean()

    # seen average
    mask = np.zeros_like(res)
    for i in range(len(order)):
        mask += np.eye(11, k=-(i))

    seen_average = (mask * res).sum(axis=0) / mask.sum(axis=0)    
    tol_seen_average = seen_average.mean()

    save_dir, _ = os.path.split(args.file_path)

    if 'CIL' in args.file_path:
        save_dir = os.path.join(save_dir, 'CIL_metric.csv')
    elif 'TIL' in args.file_path:
        save_dir = os.path.join(save_dir, 'TIL_metric.csv')
    
    res = pd.read_csv(args.file_path)
    metric = res.drop(index=res.index)

    
    for id, mc in zip(['average', 'last', 'transfer', 'seen_average'], [avg, last, transfer, seen_average]):
        metric.loc[len(metric.index)] = [id, ''] + (mc.tolist()) if id == 'transfer' else [id] + (mc.tolist())
    

    metric.loc[len(metric.index), list(metric)[:2]]  = ['tol_transfer', tol_transfer]
    metric.loc[len(metric.index), list(metric)[:2]]  = ['tol_average', tol_avg]
    metric.loc[len(metric.index), list(metric)[:2]]  = ['tol_last', tol_last]
    metric.loc[len(metric.index), list(metric)[:2]]  = ['tol_seen_average', tol_seen_average]

    metric.to_csv(save_dir, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='ckpt_10/exp_baseline_prompt_voc/CIL_results.csv')
    parser.add_argument("--Order", type=int, default=0)
    args = parser.parse_args()
    main(args)