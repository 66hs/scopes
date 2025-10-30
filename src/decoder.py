"""
Decoder Module

Neural network to decode PCA coordinates to gene expression.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import anndata as ad
from typing import Tuple, Optional


class ExpressionDecoder(nn.Module):
    """MLP decoder: PCA space → gene expression space."""
    
    def __init__(self, n_pcs: int, n_genes: int, hidden_dim: int = 256, dropout: float = 0.2):
        """
        Parameters
        ----------
        n_pcs : int
            Number of PCA dimensions (input)
        n_genes : int
            Number of genes (output)
        hidden_dim : int
            Hidden layer dimension
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.n_pcs = n_pcs
        self.n_genes = n_genes
        
        self.net = nn.Sequential(
            nn.Linear(n_pcs, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes)
        )
        
    def forward(self, x_pca: torch.Tensor) -> torch.Tensor:
        """
        Decode PCA coordinates to gene expression.
        
        Parameters
        ----------
        x_pca : torch.Tensor
            PCA coordinates (batch_size, n_pcs)
            
        Returns
        -------
        x_expr : torch.Tensor
            Gene expression predictions (batch_size, n_genes)
        """
        return self.net(x_pca)


def train_decoder(adata: ad.AnnData,
                 n_epochs: int = 50,
                 batch_size: int = 128,
                 learning_rate: float = 1e-3,
                 validation_split: float = 0.1,
                 verbose: bool = True) -> ExpressionDecoder:
    """
    Train expression decoder on single-cell data.
    
    Parameters
    ----------
    adata : AnnData
        Data with X_pca and normalized expression
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate
    validation_split : float
        Fraction of data for validation
    verbose : bool
        Print training progress
        
    Returns
    -------
    decoder : ExpressionDecoder
        Trained decoder model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Training decoder on {device}")
    
    # Extract data
    X_pca = adata.obsm['X_pca']
    X_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    n_pcs = X_pca.shape[1]
    n_genes = X_expr.shape[1]
    
    # Split data
    n_samples = X_pca.shape[0]
    n_val = int(n_samples * validation_split)
    
    indices = np.random.permutation(n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    X_pca_train = torch.FloatTensor(X_pca[train_idx]).to(device)
    X_expr_train = torch.FloatTensor(X_expr[train_idx]).to(device)
    
    X_pca_val = torch.FloatTensor(X_pca[val_idx]).to(device)
    X_expr_val = torch.FloatTensor(X_expr[val_idx]).to(device)
    
    # Create model
    decoder = ExpressionDecoder(n_pcs, n_genes).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_count = 0
    
    if verbose:
        print(f"Training on {len(train_idx)} cells, validating on {len(val_idx)}")
    
    for epoch in range(n_epochs):
        # Training
        decoder.train()
        train_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(train_idx), batch_size):
            batch_end = min(i + batch_size, len(train_idx))
            batch_idx = np.arange(i, batch_end) - i  # Local indices
            
            X_batch = X_pca_train[i:batch_end]
            y_batch = X_expr_train[i:batch_end]
            
            optimizer.zero_grad()
            y_pred = decoder(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        
        # Validation
        decoder.eval()
        with torch.no_grad():
            y_pred_val = decoder(X_pca_val)
            val_loss = criterion(y_pred_val, X_expr_val).item()
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    if verbose:
        print(f"✓ Decoder training complete (best val loss: {best_val_loss:.6f})")
    
    return decoder


def decode_trajectory(decoder: ExpressionDecoder,
                     trajectory_pca: np.ndarray) -> np.ndarray:
    """
    Decode PCA trajectory to gene expression trajectory.
    
    Parameters
    ----------
    decoder : ExpressionDecoder
        Trained decoder model
    trajectory_pca : np.ndarray
        Trajectory in PCA space (n_timepoints, n_cells, n_pcs)
        
    Returns
    -------
    trajectory_expr : np.ndarray
        Trajectory in gene expression space (n_timepoints, n_cells, n_genes)
    """
    device = next(decoder.parameters()).device
    decoder.eval()
    
    n_time, n_cells, n_pcs = trajectory_pca.shape
    
    # Flatten time and cell dimensions
    trajectory_flat = trajectory_pca.reshape(-1, n_pcs)
    traj_tensor = torch.FloatTensor(trajectory_flat).to(device)
    
    # Decode
    with torch.no_grad():
        expr_flat = decoder(traj_tensor).cpu().numpy()
    
    n_genes = expr_flat.shape[1]
    trajectory_expr = expr_flat.reshape(n_time, n_cells, n_genes)
    
    return trajectory_expr


def compute_reconstruction_error(decoder: ExpressionDecoder,
                                 adata: ad.AnnData,
                                 sample_size: Optional[int] = None) -> float:
    """
    Compute reconstruction error on validation set.
    
    Parameters
    ----------
    decoder : ExpressionDecoder
        Trained decoder
    adata : AnnData
        Data to validate on
    sample_size : int, optional
        Number of samples to use (default: all)
        
    Returns
    -------
    rmse : float
        Root mean squared error
    """
    device = next(decoder.parameters()).device
    decoder.eval()
    
    X_pca = adata.obsm['X_pca']
    X_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    if sample_size is not None and sample_size < X_pca.shape[0]:
        idx = np.random.choice(X_pca.shape[0], sample_size, replace=False)
        X_pca = X_pca[idx]
        X_expr = X_expr[idx]
    
    X_pca_t = torch.FloatTensor(X_pca).to(device)
    X_expr_t = torch.FloatTensor(X_expr).to(device)
    
    with torch.no_grad():
        X_pred = decoder(X_pca_t)
        mse = ((X_pred - X_expr_t) ** 2).mean().item()
    
    rmse = np.sqrt(mse)
    return rmse


if __name__ == "__main__":
    # Demo
    from ingest import create_demo_dataset
    from preprocess import Preprocessor
    from velocity import compute_velocity_scvelo
    
    print("Creating demo dataset...")
    adata = create_demo_dataset(n_cells=300, n_genes=500)
    
    print("Preprocessing...")
    prep = Preprocessor(n_top_genes=300, n_pcs=20)
    adata = prep.run_pipeline(adata)
    
    print("Computing velocity...")
    adata = compute_velocity_scvelo(adata, mode='stochastic')
    
    print("\nTraining decoder...")
    decoder = train_decoder(adata, n_epochs=50, batch_size=64)
    
    print("\nEvaluating decoder...")
    rmse = compute_reconstruction_error(decoder, adata)
    print(f"Reconstruction RMSE: {rmse:.6f}")
    
    print("\nTesting trajectory decoding...")
    from engine_explicit_field import simulate_trajectory
    
    def vf(x):
        return -x * 0.1  # Simple attractor
    
    x0 = adata.obsm['X_pca'][:5]
    traj_pca = simulate_trajectory(vf, x0, t_max=5, n_steps=10)
    traj_expr = decode_trajectory(decoder, traj_pca)
    
    print(f"✓ Decoded trajectory shape: {traj_expr.shape}")
    print(f"  Expected: (10, 5, 300) for (time, cells, genes)")
    
    print("\n✓ Decoder module works!")
