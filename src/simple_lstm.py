import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.evaluations import calc_nondiag_score

class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # return (torch.normal(0, 1, size=(1, self.hidden_size)).to('cuda'),
        # torch.normal(0, 1, size=(1, self.hidden_size)).to('cuda'))

        return (torch.zeros(1, self.hidden_size).to('cuda'),
                torch.zeros(1, self.hidden_size).to('cuda'))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        lstm_out = F.relu6(lstm_out)
        y_pred = self.linear(lstm_out)
        # y_pred = F.normalize(y_pred, dim=1)
        return y_pred

class LinearNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearNet, self).__init__()
        self.hidden_size = hidden_size
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = F.leaky_relu(y_pred)
        y_pred = self.out(y_pred)
        # y_pred = F.normalize(y_pred, dim=1)
        y_pred = F.sigmoid(y_pred)
        return y_pred

class AttentionNet(torch.nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size):
        super(AttentionNet, self).__init__()
        self.hidden_size = hidden_size
        # self.linear = torch.nn.Linear(input_size, hidden_size)
        # self.hidden_size = seq_len
        self.seq_len = seq_len
        self.out = torch.nn.Linear(hidden_size * input_size, output_size)
        self.out2 = torch.nn.Linear(output_size, output_size)

        self.Q = Parameter(torch.Tensor(seq_len, hidden_size))
        # self.K = Parameter(torch.Tensor(seq_len, hidden_size))
        # self.V = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.K = self.Q
        self.V = self.Q

        self.layer_norm = torch.nn.LayerNorm(input_size)
        # self.layer_norm = torch.nn.LayerNorm(seq_len)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.Q.size(1))
        self.Q.data.uniform_(-stdv, stdv)
        self.K.data.uniform_(-stdv, stdv)
        self.V.data.uniform_(-stdv, stdv)

    def calc_QK(self, x):
        Q = torch.matmul(x, self.Q)
        K = torch.matmul(x, self.K)
        V = torch.matmul(x, self.V)

        QK = torch.matmul(Q, K.permute(0, 2, 1))
        QK = QK / np.sqrt(self.hidden_size)
        QK = F.softmax(QK, dim=-1)
        # QK = F.sigmoid(QK)
        return QK, V
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        # x = x.permute(0, 2, 1)

        # x = self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x = self.layer_norm(x)

        # x = x + 0.5

        # x = x / x.sum(dim=2, keepdim=True)

        # gm = torch.exp(torch.mean(torch.log(x), dim=2, keepdim=True))
        # x = x / gm

        QK, V = self.calc_QK(x)
        return QK

        # y_pred = torch.matmul(QK, V)
        # y_pred += x
        # y_pred = y_pred.reshape(y_pred.shape[0], -1)
        # y_pred = self.out(y_pred)
        # y_pred = F.sigmoid(y_pred)
        # y_pred = F.normalize(y_pred, dim=1)
        # y_pred = F.softmax(y_pred, dim=1)
        # return y_pred, QK

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

input_size = 50
seq_len = 100
hidden_size = 1024
output_size = input_size

abundance = np.load('d:\\microbial_network\\microbial_network_explore\\data\\abundance.npy')
adj = np.load('d:\\microbial_network\\microbial_network_explore\\data\\adj.npy')
adj_norm = adj / adj.sum(axis=1, keepdims=True)

# def construct_batch(abundance, batch_size):

abd_test = torch.from_numpy(abundance[[-1], :, :seq_len]).float().to('cuda')

abundance = torch.from_numpy(abundance[:-2, :, :]).float().to('cuda')
adj_norm = torch.from_numpy(adj_norm).float().to('cuda')

prauc = []
roauc = [] 

model = AttentionNet(input_size, seq_len, hidden_size, output_size).to('cuda')

# abd = torch.from_numpy(abundance[[0], :, :100]).float().to('cuda')
# model(abd)

loss_fn = torch.nn.MSELoss().to('cuda')
# loss_fn = torch.nn.BCELoss().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)#, weight_decay=1e-2)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5, verbose=False)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-5, verbose=False)

epoch_num = 500
abundance_shape0, abundance_shape2 = abundance.shape[0], abundance.shape[2]

abundance = abundance + 0.5
abundance = abundance / abundance.sum(dim=2, keepdim=True)
gm = torch.exp(torch.mean(torch.log(abundance), dim=2, keepdim=True))
abundance = torch.log(abundance / gm)

for epoch in range(epoch_num):
    # Construct batch
    idx = np.arange(abundance_shape2 - seq_len - 1)
    idx = np.tile(idx, (abundance_shape0, 1))
    idx = shuffle_along_axis(idx, 1)

    # shuffle sample
    sample_idx = np.arange(abundance_shape0)
    np.random.shuffle(sample_idx)

    # Expand sample_idx
    sample_idx_expanded = sample_idx[:, None, None]

    for i in range(abundance_shape2 - seq_len - 1):
        # # Calculate the start and end indices for slicing
        # start_idx = idx[:, i:i+1]
        # end_idx = start_idx + seq_len
        # # Calculate the index for the y tensor
        # y_idx = end_idx + 1
        # # Create meshgrid for advanced indexing
        # sample_idx_grid, start_idx_grid = np.meshgrid(sample_idx, start_idx, indexing='ij')
        # _, end_idx_grid = np.meshgrid(sample_idx, end_idx, indexing='ij')
        # _, y_idx_grid = np.meshgrid(sample_idx, y_idx, indexing='ij')

        # # Perform advanced indexing using the meshgrids
        # x = abundance[sample_idx_grid, :, start_idx_grid:end_idx_grid]
        # y = abundance[sample_idx_grid, :, y_idx_grid]

        # Use list comprehensions to construct x and y
        x = [abundance[k, :, idx[k, i]:idx[k, i] + seq_len] for k in sample_idx]
        y = [abundance[k, :, idx[k, i] + seq_len + 1] for k in sample_idx]
        # Move the tensors to GPU (if available)
        x = torch.stack(x).to('cuda')
        y = torch.stack(y).to('cuda')
        y_adj = adj_norm[sample_idx, :, :]

        # x = []
        # y = []
        # for j in range(abundance.shape[0]):
        #     k = sample_idx[j]
        #     x.append(abundance[k, :, idx[k, i]:idx[k, i]+seq_len])
        #     y.append(abundance[k, :, idx[k, i]+seq_len+1])
        # x = torch.stack(x).to('cuda')
        # y = torch.stack(y).to('cuda')
        # y_adj = adj_norm[sample_idx, :, :]

        # x = torch.from_numpy(abundance[:, :, i:i+seq_len]).float().to('cuda')
        # y = torch.from_numpy(abundance[:, :, i+seq_len+1]).float().to('cuda')
        # y_pred, QK = model(x)
        QK = model(x)
        # loss = loss_fn(y_pred, y)
        loss_adj = loss_fn(QK, y_adj)
        # loss = loss + loss_adj
        loss = loss_adj
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     print(loss.item())
        scheduler.step()

    print('epoch: ', epoch, 'loss: ', loss.item())
    # sample = abundance[[0], :, :seq_len]
    sample = abd_test
    # sample = torch.from_numpy(abundance[[0], :, :seq_len]).float().to('cuda')
    with torch.no_grad():
        adj_pred= model.calc_QK(sample)[0].detach().cpu().numpy().squeeze()# * adj[0, :, :].sum(axis=1, keepdims=True)

    pr, ro = calc_nondiag_score(adj_pred, adj[-1, :, :], metrics=[average_precision_score, roc_auc_score], rowwise=True)
    print('prauc: ', pr, 'roauc: ', ro)

# for idx in range(abundance.shape[0]):
#     model = SimpleLSTM(input_size, hidden_size, output_size).to('cuda')
#     # model = LinearNet(input_size, hidden_size, output_size).to('cuda')
#     loss_fn = torch.nn.MSELoss().to('cuda')
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     for epoch in range(1000):
#         for i in range(abundance.shape[2]-11):
#             x = torch.from_numpy(abundance[[idx], :, i:i+10]).float().to('cuda').permute(2, 1, 0).squeeze()
#             y = torch.from_numpy(abundance[[idx], :, i+1:i+11]).float().to('cuda').permute(2, 1, 0).squeeze()
#         # x = torch.from_numpy(abundance[[idx], :, :-1]).float().to('cuda').permute(2, 1, 0).squeeze()
#         # y = torch.from_numpy(abundance[[idx], :, 1:]).float().to('cuda').permute(2, 1, 0).squeeze()
#             y_pred = model(x)
#             loss = loss_fn(y_pred, y)
#             print(epoch, loss.item(), end='\r')
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#     print('')
#     input = torch.zeros(input_size, input_size).float().to('cuda')
#     input.fill_diagonal_(1)
#     output = model(input)

#     # pr = average_precision_score(adj[0, :, :].ravel(), np.abs(output.detach().cpu().numpy().ravel()))
#     # ro = roc_auc_score(adj[0, :, :].ravel(), np.abs(output.detach().cpu().numpy().ravel()))
#     pr, ro = calc_nondiag_score(output.detach().cpu().numpy(), adj[idx, :, :], metrics=[average_precision_score, roc_auc_score])
#     print(f"PR AUC: {pr}, ROC AUC: {ro}")
#     prauc.append(pr)
#     roauc.append(ro)
