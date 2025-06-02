import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.sparse as sp

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """       
    def __init__(self, in_channels, out_channels, orders, activation='relu'): 
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)
        
    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)
        
    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = []
        supports.append(A_q)
        supports.append(A_h)
        
        x0 = X.permute(1, 2, 0)  # (num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
                
        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])         
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)     
        x += self.bias
        
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)   
            
        return x

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

class DGCN(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets
   x_s
    |GNN_3
   H_2 + H_1
    |GNN_2
   H_1
    |GNN_1
  x^y_m     
    """
    def __init__(self, h, y, z, w, k): 
        super(DGCN, self).__init__()
        self.time_dimension = h
        self.hidden_dimension1 = y
        self.hidden_dimension2 = z
        self.hidden_dimension3 = w
        self.order = k

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension1, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension1, self.hidden_dimension2, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension2, self.hidden_dimension3, self.order)

        self.GNN4 = D_GCN(self.hidden_dimension3, self.hidden_dimension2, self.order)
        self.GNN5 = D_GCN(self.hidden_dimension2, self.hidden_dimension1, self.order)
        self.GNN6 = D_GCN(self.hidden_dimension1, self.time_dimension, self.order, activation='linear')

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """  
        X_S = X.permute(0, 2, 1)  # to correct the input dims 
        
        X_s1 = self.GNN1(X_S, A_q, A_h)  # y
        X_s2 = self.GNN2(X_s1, A_q, A_h)  # z
        X_s3 = self.GNN3(X_s2, A_q, A_h)  # w
        
        X_s4 = self.GNN4(X_s3, A_q, A_h) + X_s2  # z
        X_s5 = self.GNN5(X_s4, A_q, A_h) + X_s1  # y
        X_s6 = self.GNN6(X_s5, A_q, A_h) + X_S  # h

        X_res = X_s6.permute(0, 2, 1)
               
        return X_res

def loss_function(pred, true):
    """
    Calculate combined loss with MSE and KL divergence
    Args:
        pred: Model predictions
        true: Ground truth values
    """
    # MSE Loss
    mse_loss = nn.MSELoss()(pred.reshape(-1), true.reshape(-1))
    
    # KL Divergence Loss
    kl_loss = F.kl_div(
        F.log_softmax(pred, dim=1),
        F.softmax(true, dim=1),
        reduction='batchmean'
    )
    
    # Combined loss with weight
    total_loss = mse_loss + 0.01 * kl_loss
    
    return total_loss 