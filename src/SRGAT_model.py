# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
@adopted by: Yufu li on May 06, 2023
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv
import numpy as np


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, all_item_embedding, batch):
        sections = torch.bincount(batch)
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))  # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in
                           v_i)  # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))  # |V|_i * 1
        s_g_whole = alpha * session_embedding  # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))  # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))

        # Eq(8)
        z_i_hat = torch.mm(s_h, all_item_embedding.weight.transpose(1, 0))

        return z_i_hat


class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """

    def __init__(self, hidden_size, n_node):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gated = GatedGraphConv(self.hidden_size, num_layers=1)
        self.e2s = Embedding2Score(self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch = data.x - 1, data.edge_index, data.batch

        embedding = self.embedding(x).squeeze()
        hidden = self.gated(embedding, edge_index)
        hidden2 = F.relu(hidden)

        return self.e2s(hidden2, self.embedding, batch)


def forward(model, loader, device, writer, epoch, top_k=20, optimizer=None, train_flag=True):
    if train_flag:
        model.train()
    else:
        model.eval()
        hit, mrr = [], []

    mean_loss = 0.0
    updates_per_epoch = len(loader)

    for i, batch in enumerate(loader):
        if train_flag:
            optimizer.zero_grad()
        scores = model(batch.to(device))
        targets = batch.y - 1
        loss = model.loss_function(scores, targets)

        if train_flag:
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)
        else:
            sub_scores = scores.topk(top_k)[1]  # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))

        mean_loss += loss / batch.num_graphs

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        writer.add_scalar('index/hit', hit, epoch)
        writer.add_scalar('index/mrr', mrr, epoch)