"""
Vector Field Interpolation

Fit smooth vector fields from discrete velocity estimates.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import RBFInterpolator
from typing import Callable, Tuple
import warnings


class VectorFieldInterpolator:
    """
    Interpolate vector fields from discrete observations.
    
    Supports multiple interpolation methods:
    - kNN+RBF: Fast, local interpolation
    - Global RBF: Smooth, global interpolation
    """
    
    def __init__(self,
                 method: str = 'knn_rbf',
                 n_neighbors: int = 30,
                 rbf_kernel: str = 'thin_plate_spline'):
        """
        Parameters
        ----------
        method : str
            Interpolation method: 'knn_rbf' or 'rbf'
        n_neighbors : int
            Number of neighbors for kNN
        rbf_kernel : str
            RBF kernel type
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.rbf_kernel = rbf_kernel
        
        # Will be filled after fitting
        self.X_train = None
        self.V_train = None
        self.knn = None
        self.rbf_models = None
    
    def fit(self, X: np.ndarray, V: np.ndarray) -> None:
        """
        Fit interpolator to velocity data.
        
        Parameters
        ----------
        X : np.ndarray
            Training positions (n_samples, n_dims)
        V : np.ndarray
            Training velocities (n_samples, n_dims)
        """
        X = np.asarray(X, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        
        # Remove NaN rows
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(V).any(axis=1))
        X = X[valid_mask]
        V = V[valid_mask]
        
        if X.shape[0] == 0:
            raise ValueError("No valid training data after removing NaN values")
        
        self.X_train = X
        self.V_train = V
        
        if self.method == 'knn_rbf':
            self._fit_knn_rbf()
        elif self.method == 'rbf':
            self._fit_global_rbf()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _fit_knn_rbf(self) -> None:
        """Fit kNN+RBF interpolator."""
        self.knn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, self.X_train.shape[0]),
            algorithm='ball_tree'
        )
        self.knn.fit(self.X_train)
    
    def _fit_global_rbf(self) -> None:
        """Fit global RBF interpolator."""
        self.rbf_models = []
        for dim in range(self.V_train.shape[1]):
            rbf = RBFInterpolator(
                self.X_train,
                self.V_train[:, dim],
                kernel=self.rbf_kernel,
                epsilon=1.0
            )
            self.rbf_models.append(rbf)
    
    def __call__(self, X_query: np.ndarray) -> np.ndarray:
        """
        Predict velocities at query points.
        
        Parameters
        ----------
        X_query : np.ndarray
            Query positions (n_query, n_dims)
            
        Returns
        -------
        V_pred : np.ndarray
            Predicted velocities (n_query, n_dims)
        """
        if self.method == 'knn_rbf':
            return self._predict_knn_rbf(X_query)
        elif self.method == 'rbf':
            return self._predict_global_rbf(X_query)
    
    def _predict_knn_rbf(self, X_query: np.ndarray) -> np.ndarray:
        """
        Predict velocity using kNN + RBF interpolation.
        
        FIX: Handles NaN values and infinities
        
        Parameters
        ----------
        X_query : np.ndarray
            Query points (n_query, n_dims)
            
        Returns
        -------
        V_pred : np.ndarray
            Predicted velocities (n_query, n_dims)
        """
        X_query = np.asarray(X_query, dtype=np.float32)
        
        # CRITICAL FIX: Handle NaN and infinite values
        nan_mask = np.isnan(X_query)
        if np.any(nan_mask):
            X_query = np.nan_to_num(X_query, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clip to reasonable range
        X_query = np.clip(X_query, -1e6, 1e6)
        
        # Ensure no NaN in training data
        if np.any(np.isnan(self.X_train)):
            warnings.warn("NaN detected in training data X")
            self.X_train = np.nan_to_num(self.X_train, nan=0.0)
        
        if np.any(np.isnan(self.V_train)):
            warnings.warn("NaN detected in training data V")
            self.V_train = np.nan_to_num(self.V_train, nan=0.0)
        
        # Predict
        distances, indices = self.knn.kneighbors(X_query)
        
        # RBF interpolation with safety checks
        V_pred = np.zeros((X_query.shape[0], self.V_train.shape[1]), dtype=np.float32)
        
        for i in range(X_query.shape[0]):
            # Get nearby points
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            # Avoid division by zero
            neighbor_distances = np.maximum(neighbor_distances, 1e-8)
            
            # RBF weights (inverse distance)
            weights = 1.0 / neighbor_distances
            weights = weights / np.sum(weights)  # Normalize
            
            # Weighted average
            V_pred[i] = np.sum(
                self.V_train[neighbor_indices] * weights[:, None],
                axis=0
            )
        
        # Final safety check
        V_pred = np.nan_to_num(V_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        V_pred = np.clip(V_pred, -1e6, 1e6)
        
        return V_pred
    
    def _predict_global_rbf(self, X_query: np.ndarray) -> np.ndarray:
        """Predict using global RBF models."""
        X_query = np.asarray(X_query, dtype=np.float32)
        
        # Handle NaN
        X_query = np.nan_to_num(X_query, nan=0.0, posinf=1e6, neginf=-1e6)
        X_query = np.clip(X_query, -1e6, 1e6)
        
        V_pred = np.zeros((X_query.shape[0], len(self.rbf_models)), dtype=np.float32)
        
        for dim, rbf_model in enumerate(self.rbf_models):
            V_pred[:, dim] = rbf_model(X_query)
        
        # Final safety check
        V_pred = np.nan_to_num(V_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        V_pred = np.clip(V_pred, -1e6, 1e6)
        
        return V_pred
    
    def compute_divergence(self, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Compute vector field divergence via finite differences.
        
        Parameters
        ----------
        X : np.ndarray
            Query points (n_samples, n_dims)
        eps : float
            Finite difference step size
            
        Returns
        -------
        div : np.ndarray
            Divergence at query points (n_samples,)
        """
        div = np.zeros(X.shape[0])
        
        for dim in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, dim] += eps
            V_plus = self(X_plus)
            
            X_minus = X.copy()
            X_minus[:, dim] -= eps
            V_minus = self(X_minus)
            
            div += (V_plus[:, dim] - V_minus[:, dim]) / (2 * eps)
        
        return div


def fit_vector_field_from_scvelo(adata, method='knn_rbf', basis='pca', n_neighbors=30):
    """
    Fit vector field from scVelo velocity estimates.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with velocity estimates
    method : str
        Interpolation method
    basis : str
        Basis to use ('pca', 'umap', etc.)
    n_neighbors : int
        Number of neighbors for kNN
        
    Returns
    -------
    vf : VectorFieldInterpolator
        Fitted vector field
    """
    # Get positions
    X_key = f'X_{basis}'
    if X_key not in adata.obsm:
        raise ValueError(f"{X_key} not found in adata.obsm")
    
    X = adata.obsm[X_key]
    
    # Get velocities
    V_key = f'velocity_{basis}'
    if V_key not in adata.obsm:
        raise ValueError(f"{V_key} not found in adata.obsm")
    
    V = adata.obsm[V_key]
    
    # Fit interpolator
    vf = VectorFieldInterpolator(method=method, n_neighbors=n_neighbors)
    vf.fit(X, V)
    
    return vf


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    V_train = -X_train + 0.1 * np.random.randn(100, 5)
    
    # Fit
    vf = VectorFieldInterpolator(method='knn_rbf')
    vf.fit(X_train, V_train)
    
    # Predict
    X_query = np.random.randn(10, 5)
    V_pred = vf(X_query)
    
    print(f"✓ Vector field works!")
    print(f"  Input shape: {X_query.shape}")
    print(f"  Output shape: {V_pred.shape}")
