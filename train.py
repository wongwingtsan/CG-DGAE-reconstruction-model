import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

from config.default_config import Config
from models.cg_dgae import DGCN, calculate_random_walk_matrix
from utils.data_utils import (
    load_station_data,
    create_adjacency_matrix,
    load_time_series_data,
    normalize_data
)
from utils.data_augmentation import corrupt_batch, insertsection

def parse_args():
    parser = argparse.ArgumentParser(description='Train DGCN model')
    parser.add_argument('--config', type=str, default='config/default_config.py',
                      help='Path to config file')
    # Data parameters
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--section_prob', type=float, help='Probability of section corruption')
    parser.add_argument('--point_prob', type=float, help='Probability of point corruption')
    parser.add_argument('--zero_prob', type=float, help='Probability of zero corruption')
    # Model parameters
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    return parser.parse_args()

def train_model(config, training_, cluster, adj, device):
    """
    Train the DGCN model following the original implementation
    """
    # Initialize model
    DGCN_model = DGCN(
        h=config.time_dimension,
        y=config.hidden_dimension1,
        z=config.hidden_dimension2,
        w=config.hidden_dimension3,
        k=config.order
    ).to(device)
    
    # Setup training
    optimizer = optim.Adam(DGCN_model.parameters(), lr=config.learning_rate)
    losslist = []
    
    # Training loop
    for epoch in range(config.num_epochs):
        loss_epoch = 0
        DGCN_model.train()
        
        # Train on each cluster
        for j in list(set(cluster['cluster'].values)):
            CLUSTER = cluster.loc[cluster['cluster'] == j]
            know_nodes = list(CLUSTER.index)
            clustered_nodes = [training_[n] for n in know_nodes]
            
            X_ = np.array(clustered_nodes).astype(np.float32)
            training_set = X_.transpose()
            E_maxvalue = training_set.max()
            label_set = X_.transpose().copy()
            
            # Process batches
            for i in range(int(training_set.shape[0] // (config.time_dimension * config.batch_size))):
                # Generate random time points
                t_random = np.random.randint(
                    0,
                    high=(training_set.shape[0] - config.time_dimension),
                    size=config.batch_size,
                    dtype='l'
                )
                
                feed_batch_tr = []
                feed_batch_lb = []
                
                # Apply corruption to some samples
                num_error_batches = int(config.batch_size * config.corrupted_ratio)
                error_batch_indices = random.sample(range(config.batch_size), num_error_batches)
                
                for q in range(config.batch_size):
                    if q in error_batch_indices:
                        node = random.randint(0, len(know_nodes) - 1)
                        training_set[t_random[q]:t_random[q] + config.time_dimension, node] = \
                            insertsection(training_set[t_random[q]:t_random[q] + config.time_dimension, :].transpose()[node])
                    
                    feed_batch_tr.append(training_set[t_random[q]:t_random[q] + config.time_dimension, :])
                    feed_batch_lb.append(label_set[t_random[q]:t_random[q] + config.time_dimension, :])
                
                # Prepare inputs
                inputs = np.array(feed_batch_tr)
                labels = np.array(feed_batch_lb)
                Mf_inputs = inputs / E_maxvalue
                Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
                
                # Prepare adjacency matrices
                A_dynamic = adj[list(know_nodes), :][:, list(know_nodes)]
                A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
                A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)
                
                # Prepare labels
                node_label = torch.from_numpy(labels).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                X_res = DGCN_model(Mf_inputs, A_q, A_h)
                X_res = X_res * E_maxvalue
                
                # Calculate losses
                loss_ = nn.MSELoss()(X_res.reshape(-1), node_label.reshape(-1))
                KL_loss = F.kl_div(
                    F.log_softmax(X_res, dim=1),
                    F.softmax(node_label, dim=1),
                    reduction='batchmean'
                )
                loss = loss_ + 0.01 * KL_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                loss_epoch += loss
        
        print('the loss is {} at epoch {}'.format(loss, epoch))
        losslist.append(loss.detach().cpu().numpy())
    
    return DGCN_model, losslist

def main():
    # Parse arguments and update config
    args = parse_args()
    config = Config()
    config.update(**{k: v for k, v in vars(args).items() if v is not None})
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    station_names, cluster_df = load_station_data(config)
    adj_matrix = create_adjacency_matrix(station_names, config.distance_matrix_file, config)
    train_data, test_data = load_time_series_data(station_names, config)
    
    # Normalize data
    train_data, min_val, max_val = normalize_data(train_data)
    test_data = (test_data - min_val) / (max_val - min_val)
    
    # Move to device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    adj_matrix = adj_matrix.to(device)
    
    # Train model
    model, losslist = train_model(config, train_data, cluster_df, adj_matrix, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'losslist': losslist,
        'min_val': min_val,
        'max_val': max_val,
    }, f"{config.checkpoint_dir}/best_model.pt")

if __name__ == "__main__":
    main() 