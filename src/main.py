#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
@adopted by: Yufu li on May 06, 2023

@inproceedings{Wu:2019ke,
title = {{Session-based Recommendation with Graph Neural Networks}},
author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
year = 2019,
booktitle = {Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence},
location = {Honolulu, HI, USA},
month = jul,
volume = 33,
number = 1,
series = {AAAI '19},
pages = {346--353},
url = {https://aaai.org/ojs/index.php/AAAI/article/view/3804},
doi = {10.1609/aaai.v33i01.3301346},
editor = {Pascal Van Hentenryck and Zhi-Hua Zhou},
}
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from tensorboardX import SummaryWriter
import os
import logging
import datetime
import numpy as np
import torch
from tqdm import tqdm
from datasets import MultiSessionsGraph
from torch_geometric.data import DataLoader



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--model', default='GNN',
                    help='Select from different models to do experiments, selections: GNN, GCSAN, SRGAT')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dynamic', type=bool, default=False)
parser.add_argument('--device', type=str, default='mps', help='Define the device for model to run, typically cuda, cpu and mps')
parser.add_argument('--topk', type=int, default=20, help='top K indicator for evaluation')
opt = parser.parse_args()
logging.info(opt)


assert opt.model in ['GNN', 'GCSAN', 'SRGAT'], f"The selection {opt.model} is not one of GNN, GCSAN, SRGAT"
if opt.model == 'GNN':
    from model import *
elif opt.model == 'GCSAN':
    from GCSAN_model import *
else: # opt.model == 'SRGAT'
    from SRGAT_model import Embedding2Score, GNNModel, forward
# else:

cur_dir = os.getcwd()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
                    filename=cur_dir + '/../log/' + str(opt.dataset) + '/' + str(opt.model))
log_dir = cur_dir + '/../log/' + str(opt.dataset) + '/' + str(opt)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.warning('logging to {}'.format(log_dir))
writer = SummaryWriter(log_dir)


def train_test(model, train_data, test_data, writer, epoch, topk=opt.topk):
    logging.info('start training: {}'.format(datetime.datetime.now()))
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        updates_per_epoch = len(slices)
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            logging.info('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
        writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + j)
    logging.info('\tLoss:\t%.3f' % total_loss)
    writer.add_scalar('loss/train_loss', total_loss/len(slices), epoch)

    logging.info('start predicting: {}'.format(datetime.datetime.now()))
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(topk)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    # move scheduler after optimizer
    model.scheduler.step()
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    writer.add_scalar('loss/test_loss', total_loss / len(slices), epoch)
    writer.add_scalar('index/hit', hit, epoch)
    writer.add_scalar('index/mrr', mrr, epoch)
    return hit, mrr


def main():
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
        # n_node = 227
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    if opt.model in [ 'GNN', 'GCSAN']:
        # Data loading
        train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
        if opt.validation:
            train_data, valid_data = split_validation(train_data, opt.valid_portion)
            test_data = valid_data
        else:
            test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
        train_data = Data(train_data, shuffle=True, opt=opt)
        test_data = Data(test_data, shuffle=False, opt=opt)

        # Model training
        if opt.model == 'GNN':
            model = trans_to_cuda(SessionGraph(opt, n_node))
        elif opt.model == 'GCSAN':
            model = trans_to_cuda(SessionGraph(opt, n_node, max(train_data.len_max, test_data.len_max)))
        start = time.time()
        best_result = [0, 0]
        best_epoch = [0, 0]
        bad_counter = 0
        for epoch in range(opt.epoch):
            logging.info('-------------------------------------------------------')
            logging.info('epoch: {}'.format(epoch))
            hit, mrr = train_test(model, train_data, test_data, writer, epoch, topk=20)
            flag = 0
            if hit >= best_result[0]:
                best_result[0] = hit
                best_epoch[0] = epoch
                flag = 1
            if mrr >= best_result[1]:
                best_result[1] = mrr
                best_epoch[1] = epoch
                flag = 1
            logging.info('Best Result:')
            logging.info('\tPrecision@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
            bad_counter += 1 - flag
            if bad_counter >= opt.patience:
                break
        logging.info('-------------------------------------------------------')
        end = time.time()
        logging.info("Run time: %f s" % (end - start))
    else:
        # data loading for SR-GAT
        train_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='train')
        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
        test_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='test')
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

        # model training
        model = GNNModel(hidden_size=opt.hiddenSize, n_node=n_node).to(opt.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        logging.warning(model)
        start = time.time()
        best_result = [0, 0]
        best_epoch = [0, 0]
        bad_counter = 0
        for epoch in tqdm(range(opt.epoch)):
            scheduler.step()
            forward(model, train_loader, opt.device, writer, epoch, top_k=opt.topk, optimizer=optimizer, train_flag=True)
            with torch.no_grad():
                hit, mrr = forward(model, test_loader, opt.device, writer, epoch, top_k=opt.topk, train_flag=False)
                flag = 0
                if hit >= best_result[0]:
                    best_result[0] = hit
                    best_epoch[0] = epoch
                    flag = 1
                if mrr >= best_result[1]:
                    best_result[1] = mrr
                    best_epoch[1] = epoch
                    flag = 1
                logging.info('Best Result:')
                logging.info('\tPrecision@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
                best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
                bad_counter += 1 - flag
                if bad_counter >= opt.patience:
                    break


if __name__ == '__main__':
    main()
