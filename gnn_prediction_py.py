"""
GNN-Based Market Prediction Module

This module implements Graph Neural Networks for predictive modeling of market returns,
leveraging the structural information in asset correlation networks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
import copy
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class GNNMarketModel(nn.Module):
    """
    Graph Neural Network for market modeling and prediction
    
    This model uses graph convolutions to learn from the structure of asset
    correlation networks for predictive modeling.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, dropout=0.5,
                gnn_type='gcn', pooling='mean'):
        """
        Initialize GNN model
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input node features
        hidden_dim : int
            Dimension of hidden layers
        output_dim : int
            Dimension of output (1 for regression, n for classification)
        num_layers : int
            Number of GNN layers
        dropout : float
            Dropout probability
        gnn_type : str
            Type of GNN layer: 'gcn', 'sage', or 'gat'
        pooling : str
            Type of graph pooling: 'mean', 'max', or 'sum'
        """
        super(GNNMarketModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.pooling = pooling
        
        # Input layer
        if gnn_type == 'gcn':
            # Graph Convolutional Network
            self.conv_first = GCNConv(input_dim, hidden_dim)
        elif gnn_type == 'sage':
            # GraphSAGE (Sample and Aggregate)
            self.conv_first = SAGEConv(input_dim, hidden_dim)
        elif gnn_type == 'gat':
            # Graph Attention Network
            self.conv_first = GATConv(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
            
        # Hidden layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
        
        # Output layer (after pooling)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, data):
        """
        Forward pass
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            Graph data object
            
        Returns:
        --------
        torch.Tensor
            Output predictions
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # First layer
        x = self.conv_first(x, edge_index, edge_weight)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Hidden layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = self.batch_norms[i+1](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Pooling
        if self.pooling == 'mean':
            x = torch_geometric.nn.global_mean_pool(x, data.batch)
        elif self.pooling == 'max':
            x = torch_geometric.nn.global_max_pool(x, data.batch)
        elif self.pooling == 'sum':
            x = torch_geometric.nn.global_add_pool(x, data.batch)
        
        # Output layer
        x = self.fc_out(x)
        
        return x


class MarketGNNPredictor:
    """
    Market predictor using GNN
    
    This class handles the data preparation, model training, and evaluation
    for predicting market returns using Graph Neural Networks.
    """
    
    def __init__(self, returns, prediction_horizon=5, lookback_window=60, feature_window=20):
        """
        Initialize predictor
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame containing asset returns
        prediction_horizon : int
            Number of days ahead to predict
        lookback_window : int
            Number of days to use for graph construction
        feature_window : int
            Number of days to use for node features
        """
        self.returns = returns
        self.prediction_horizon = prediction_horizon
        self.lookback_window = lookback_window
        self.feature_window = feature_window
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def prepare_data(self, test_ratio=0.2, val_ratio=0.1, correlation_threshold=0.3):
        """
        Prepare data for GNN training
        
        Creates graph objects for each time window and splits data into train/val/test sets.
        
        Parameters:
        -----------
        test_ratio : float
            Ratio of data to use for testing
        val_ratio : float
            Ratio of data to use for validation
        correlation_threshold : float
            Threshold for correlation to create edges
            
        Returns:
        --------
        tuple
            (train_loader, val_loader, test_loader)
        """
        print(f"Preparing data for GNN training (prediction horizon: {self.prediction_horizon} days)")
        
        # Generate target variables (future returns)
        targets = {}
        for i in range(1, self.prediction_horizon + 1):
            # Shifted returns for each prediction horizon
            shifted = self.returns.shift(-i)
            targets[f'target_{i}d'] = shifted
        
        # Combine into DataFrame
        targets_df = pd.DataFrame(index=self.returns.index)
        for col, series in targets.items():
            targets_df[col] = series
        
        # Average target across prediction horizon
        targets_df['target'] = targets_df.mean(axis=1)
        
        # Generate features and graphs
        data_list = []
        
        print("Creating graph objects...")
        for i in range(self.lookback_window, len(self.returns) - self.prediction_horizon):
            # Get window of returns for graph construction
            graph_window = self.returns.iloc[i-self.lookback_window:i]
            
            # Get smaller window for features
            feature_window = self.returns.iloc[i-self.feature_window:i]
            
            # Get date
            date = self.returns.index[i]
            
            # Get target
            target = targets_df.loc[date, 'target']
            
            # Skip if target is NaN
            if pd.isna(target):
                continue
                
            # Create graph
            graph_data = self._create_graph(graph_window, feature_window, target)
            
            # Add to list
            data_list.append(graph_data)
        
        print(f"Created {len(data_list)} graph objects")
        
        # Split data
        n_samples = len(data_list)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        # Calculate split points
        test_size = int(test_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        train_size = n_samples - test_size - val_size
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        print(f"Data split: {train_size} train, {val_size} validation, {test_size} test")
        
        # Create datasets
        train_dataset = [data_list[i] for i in train_indices]
        val_dataset = [data_list[i] for i in val_indices]
        test_dataset = [data_list[i] for i in test_indices]
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def _create_graph(self, graph_window, feature_window, target):
        """
        Create graph data object from return windows
        
        Builds a graph where nodes are assets and edges represent correlations.
        Node features are extracted from asset returns.
        """
        # Compute correlation matrix
        corr_matrix = graph_window.corr().abs()
        
        # Get node features
        node_features = []
        
        for ticker in graph_window.columns:
            # Extract features for this asset
            features = self._extract_asset_features(feature_window[ticker])
            node_features.append(features)
            
        # Convert to numpy array
        node_features = np.array(node_features)
        
        # Scale features
        node_features = self.scaler_X.fit_transform(node_features)
        
        # Create edge index and weights
        edge_index = []
        edge_weight = []
        
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                # Get correlation
                corr = corr_matrix.iloc[i, j]
                
                # Add edge if correlation is above threshold
                if corr > 0.3:
                    # Add edge in both directions (for undirected graph)
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # Add weights
                    edge_weight.append(corr)
                    edge_weight.append(corr)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        # Scale target
        target = self.scaler_y.fit_transform(np.array([[target]]))[0, 0]
        
        # Target tensor
        y = torch.tensor([target], dtype=torch.float)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        
        return data
    
    def _extract_asset_features(self, returns):
        """
        Extract features for a single asset
        
        Creates a feature vector from the asset's return time series.
        """
        features = []
        
        # Return features
        features.append(returns.mean())                   # Mean return
        features.append(returns.std())                    # Volatility
        features.append(returns.skew())                   # Skewness
        features.append(returns.kurtosis())               # Kurtosis
        
        # Add momentum features
        for window in [5, 10, 20]:
            if len(returns) >= window:
                momentum = returns.iloc[-window:].sum()   # Momentum over window
                features.append(momentum)
            else:
                features.append(0)                        # Padding
                
        # Add moving average features
        for window in [5, 10, 20]:
            if len(returns) >= window:
                ma = returns.rolling(window=window).mean().iloc[-1]
                features.append(ma)
            else:
                features.append(0)                        # Padding
                
        # Add volatility features
        for window in [5, 10, 20]:
            if len(returns) >= window:
                vol = returns.rolling(window=window).std().iloc[-1]
                features.append(vol)
            else:
                features.append(0)                        # Padding
        
        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def build_model(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, 
                   dropout=0.5, gnn_type='gcn', pooling='mean'):
        """
        Build GNN model
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input node features
        hidden_dim : int
            Dimension of hidden layers
        output_dim : int
            Dimension of output (1 for regression, n for classification)
        num_layers : int
            Number of GNN layers
        dropout : float
            Dropout probability
        gnn_type : str
            Type of GNN layer: 'gcn', 'sage', or 'gat'
        pooling : str
            Type of graph pooling: 'mean', 'max', or 'sum'
            
        Returns:
        --------
        GNNMarketModel
            Built model
        """
        print(f"Building GNN model ({gnn_type} with {num_layers} layers)")
        self.model = GNNMarketModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            pooling=pooling
        )
        
        self.model = self.model.to(self.device)
        
        return self.model
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=5e-4, 
             patience=20, verbose=True):
        """
        Train GNN model
        
        Parameters:
        -----------
        train_loader : torch_geometric.data.DataLoader
            Training data loader
        val_loader : torch_geometric.data.DataLoader
            Validation data loader
        epochs : int
            Number of epochs to train
        lr : float
            Learning rate
        weight_decay : float
            Weight decay for regularization
        patience : int
            Patience for early stopping
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        dict
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
            
        print(f"Training GNN model for {epochs} epochs (patience={patience})...")
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Initialize best validation loss for early stopping
        best_val_loss = float('inf')
        counter = 0
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = self.model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.num_graphs
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.device)
                    out = self.model(data)
                    loss = criterion(out, data.y)
                    val_loss += loss.item() * data.num_graphs
                    
            val_loss /= len(val_loader.dataset)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save best model
                best_model = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
                
            if counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
                
        # Load best model
        self.model.load_state_dict(best_model)
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        
        return history
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        test_loader : torch_geometric.data.DataLoader
            Test data loader
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained")
            
        print("Evaluating model on test data...")
        self.model.eval()
        
        # Initialize lists for predictions and targets
        preds = []
        targets = []
        
        # Evaluation loop
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.model(data)
                
                # Collect predictions and targets
                preds.append(out.cpu().numpy())
                targets.append(data.y.cpu().numpy())
                
        # Concatenate predictions and targets
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        
        # Inverse transform if scalers were used
        if hasattr(self, 'scaler_y') and self.scaler_y is not None:
            preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
            targets = self.scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))
        r2 = 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        
        # Calculate directional accuracy
        correct_dir = np.sum((preds > 0) == (targets > 0))
        dir_acc = correct_dir / len(targets)
        
        # Create metrics dictionary
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'dir_acc': dir_acc
        }
        
        print(f"Test MSE: {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print(f"Test R²: {r2:.4f}")
        print(f"Directional Accuracy: {dir_acc:.2%}")
        
        return metrics, preds, targets
    
    def predict_future(self, current_data, n_steps=5):
        """
        Predict future returns
        
        Forecasts returns for future periods using the trained GNN model.
        
        Parameters:
        -----------
        current_data : pandas.DataFrame
            Current return data
        n_steps : int
            Number of steps ahead to predict
            
        Returns:
        --------
        numpy.ndarray
            Predicted future returns
        """
        if self.model is None:
            raise ValueError("Model not built or trained")
            
        print(f"Predicting {n_steps} steps ahead...")
        self.model.eval()
        
        # Initialize predictions
        predictions = []
        
        # Make a copy of the data
        data = current_data.copy()
        
        # Prediction loop
        for step in range(n_steps):
            # Get windows
            graph_window = data.iloc[-self.lookback_window:]
            feature_window = data.iloc[-self.feature_window:]
            
            # Create graph (use 0 as dummy target)
            graph_data = self._create_graph(graph_window, feature_window, 0)
            
            # Move to device
            graph_data = graph_data.to(self.device)
            
            # Predict
            with torch.no_grad():
                pred = self.model(graph_data)
                
            # Convert to numpy
            pred = pred.cpu().numpy()[0]
            
            # Inverse transform if scalers were used
            if hasattr(self, 'scaler_y') and self.scaler_y is not None:
                pred = self.scaler_y.inverse_transform([[pred]])[0, 0]
                
            # Add to predictions
            predictions.append(pred)
            
            # Add prediction to data for next step
            new_row = pd.DataFrame([np.zeros(len(data.columns))], columns=data.columns,
                                 index=[data.index[-1] + pd.Timedelta(days=1)])
            data = pd.concat([data, new_row])
        
        return np.array(predictions)
    
    def plot_predictions(self, preds, targets, figsize=(12, 6), save_path=None):
        """
        Plot predictions vs targets
        
        Creates a scatter plot of predicted vs actual returns.
        
        Parameters:
        -----------
        preds : numpy.ndarray
            Predicted values
        targets : numpy.ndarray
            Target values
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        ax.scatter(targets, preds, alpha=0.5)
        
        # Add diagonal line (perfect predictions)
        min_val = min(np.min(targets), np.min(preds))
        max_val = max(np.max(targets), np.max(preds))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add labels and title
        ax.set_xlabel('Actual Returns')
        ax.set_ylabel('Predicted Returns')
        ax.set_title('GNN Predictions vs Actual Returns')
        
        # Add statistics
        r2 = 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_training_history(self, history, figsize=(12, 6), save_path=None):
        """
        Plot training history
        
        Creates a plot showing training and validation loss over epochs.
        
        Parameters:
        -----------
        history : dict
            Training history
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training and validation loss
        ax.plot(history['train_loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        
        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('GNN Training History')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig