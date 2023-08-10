import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F

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
        # return QK

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
    

class DNN_pred(nn.Module):
    def __init__(self, n_vertices, hidden_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_vertices = n_vertices

        self.fc1 = nn.Linear(n_vertices, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_vertices)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = x / torch.sum(x, dim=1, keepdim=True)
        # x = F.normalize(x, dim=1, p=1)
        # x = F.softmax(x, dim=1)
        return x
    

class Abs_LV(nn.Module):
    def __init__(self, n_vertices, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.g = nn.Parameter(torch.rand(n_vertices).float())
        self.interaction = nn.Parameter(torch.rand(n_vertices, n_vertices).float())

    def forward(self, x):
        interaction = torch.matmul(self.interaction, x[:, :, None]).squeeze(-1)
        inter_species = torch.mul(x, interaction)
        growth = torch.mul(x, self.g + 1)
        x = inter_species + growth
        return x
    
class Rel_LV(nn.Module):
    def __init__(self, n_vertices, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.g = nn.Parameter(torch.rand(n_vertices).float())
        self.interaction = nn.Parameter(torch.rand(n_vertices, n_vertices).float())

    def forward(self, x, steps=1):
        for i in range(steps):
            interaction = torch.matmul(self.interaction, x[i:i+1, :, None]).squeeze(-1)
            inter_species = torch.mul(x[i:i+1, :], interaction)
            growth = torch.mul(x[i:i+1, :], self.g + 1)
            x[i:i+1, :] = inter_species + growth
        y = x / torch.sum(x, dim=1, keepdim=True)
        return y


class Compo_LV(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)