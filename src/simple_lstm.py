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
        self.hidden_size = seq_len
        self.seq_len = seq_len
        self.out = torch.nn.Linear(input_size * seq_len, output_size)
        self.out2 = torch.nn.Linear(output_size, output_size)

        self.Q = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.K = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V = Parameter(torch.Tensor(hidden_size, hidden_size))

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
        return QK, V
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        # x = x.permute(0, 2, 1)

        QK, V = self.calc_QK(x)

        y_pred = torch.matmul(QK, V)
        y_pred += x
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_pred = self.out(y_pred)
        # y_pred = F.sigmoid(y_pred)
        # y_pred = F.normalize(y_pred, dim=1)
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred

input_size = 50
seq_len = 100
hidden_size = seq_len
output_size = input_size

abundance = np.load('d:\\microbial_network\\microbial_network_explore\\data\\abundance.npy')
adj = np.load('d:\\microbial_network\\microbial_network_explore\\data\\adj.npy')

# def construct_batch(abundance, batch_size):

# abundance = torch.from_numpy(abundance).float().to('cuda')

prauc = []
roauc = [] 

model = AttentionNet(input_size, 100, hidden_size, output_size).to('cuda')

# abd = torch.from_numpy(abundance[[0], :, :100]).float().to('cuda')
# model(abd)

loss_fn = torch.nn.MSELoss().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epoch_num = 500
for epoch in range(epoch_num):
    for i in range(abundance.shape[2]-seq_len-1):
        # x = abundance[:, :, i:i+seq_len]
        # y = abundance[:, :, i+seq_len+1]
        x = torch.from_numpy(abundance[:, :, i:i+seq_len]).float().to('cuda')
        y = torch.from_numpy(abundance[:, :, i+seq_len+1]).float().to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     print(loss.item())

    print('epoch: ', epoch, 'loss: ', loss.item())
    # sample = abundance[[0], :, :seq_len]
    sample = torch.from_numpy(abundance[[0], :, :seq_len]).float().to('cuda')
    adj_pred= model.calc_QK(sample)[0].detach().cpu().numpy().squeeze()

    pr, ro = calc_nondiag_score(adj_pred, adj[0, :, :], metrics=[average_precision_score, roc_auc_score])
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
