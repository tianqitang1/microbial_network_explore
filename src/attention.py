import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.misc import calc_nondiag_score
import pandas as pd


class AttentionNet(torch.nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size):
        super(AttentionNet, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.out = torch.nn.Linear(hidden_size * input_size, output_size)

         # Parameters for the attention mechanism
        self.Q = Parameter(torch.Tensor(seq_len, hidden_size)) # Query matrix
        self.K = self.Q # Key matrix
        self.V = self.Q # Value matrix

        # Layer normalization
        self.layer_norm = torch.nn.LayerNorm(input_size)

        # Initialize the parameters
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

        QK, V = self.calc_QK(x)
        return QK

        y_pred = torch.matmul(QK, V)
        # print(y_pred.shape)
        # print(x.shape)
        # y_pred += x
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_pred = self.out(y_pred)
        y_pred = F.sigmoid(y_pred)
        y_pred = F.normalize(y_pred, dim=1)
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred, QK
    
class DNN(torch.nn.Module):
    def __init__(self, n_vertices, seq_len, hidden_size, *args, **kwargs) -> None:
        super(DNN, self).__init__(*args, **kwargs)
        self.n_vertices = n_vertices
        self.hidden_size = hidden_size

        self.fc1 = torch.nn.Linear(n_vertices * seq_len, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, n_vertices * n_vertices)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.reshape(x.shape[0], self.n_vertices, self.n_vertices)

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

input_size = 50
seq_len = 500
hidden_size = 1024
output_size = input_size

abundance = np.load('d:\\microbial_network\\microbial_network_explore\\data\\abundance.npy')
adj = np.load('d:\\microbial_network\\microbial_network_explore\\data\\adj.npy')
adj_norm = adj / adj.sum(axis=1, keepdims=True)

abd_test = torch.from_numpy(abundance[[-1], :, :seq_len]).float().to('cuda')

abundance = torch.from_numpy(abundance[:-2, :, :]).float().to('cuda')
adj_norm = torch.from_numpy(adj_norm).float().to('cuda')
adj = torch.from_numpy(adj).float().to('cuda')

prauc = []
roauc = [] 

# model = AttentionNet(input_size, seq_len, hidden_size, output_size).to('cuda')
model = DNN(input_size, seq_len, hidden_size).to('cuda')

loss_fn = torch.nn.MSELoss().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=1e-2)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5, verbose=False)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-5, verbose=False)

prs = []
ros = []

epoch_num = 1000
abundance_shape0, abundance_shape2 = abundance.shape[0], abundance.shape[2]
print(abundance_shape0, abundance_shape2)
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
        # Use list comprehensions to construct x and y
        x = [abundance[k, :, idx[k, i]:idx[k, i] + seq_len] for k in sample_idx]
        # y = [abundance[k, :, idx[k, i] + seq_len + 1] for k in sample_idx]
        # Move the tensors to GPU (if available)
        x = torch.stack(x).to('cuda')
        # y = torch.stack(y).to('cuda')
        y_adj = adj[sample_idx, :, :]


        # x = torch.from_numpy(abundance[:, :, i:i+seq_len]).float().to('cuda')
        # y = torch.from_numpy(abundance[:, :, i+seq_len+1]).float().to('cuda')
        # y_pred, QK = model(x)
        # loss = loss_fn(y_pred, y)

        # QK = model(x)
        # loss_adj = loss_fn(QK, y_adj)
        # loss = loss + loss_adj
        # loss = loss_adj
        output = model(x)
        loss = loss_fn(output.reshape(y_adj.shape[0], -1), y_adj.reshape(y_adj.shape[0], -1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print('epoch: ', epoch, 'loss: ', loss.item())
    sample = abd_test
    with torch.no_grad():
        # adj_pred= model.calc_QK(sample)[0].detach().cpu().numpy().squeeze()# * adj[0, :, :].sum(axis=1, keepdims=True)
        adj_pred = model(sample).detach().cpu().numpy().squeeze()

    pr, ro = calc_nondiag_score(adj_pred, adj[-1, :, :].cpu().numpy(), metrics=[average_precision_score, roc_auc_score], rowwise=False)
    print('prauc: ', pr, 'roauc: ', ro)

    prs.append(pr)
    ros.append(ro)

    if epoch % 10 == 0:
        scores = pd.DataFrame({'prauc': prs, 'roauc': ros})
        scores.to_csv('d:\\microbial_network\\microbial_network_explore\\data\\attention_scores.csv', index=False)

    torch.save(model.state_dict(), 'd:\\microbial_network\\microbial_network_explore\\data\\attention_model.pt')
