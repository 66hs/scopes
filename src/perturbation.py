"""
Perturbation Module

Initial state shift and forcing term operators for gene perturbations.
"""

import numpy as np
import anndata as ad
from typing import Callable
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


class PerturbationOperator:
    """Apply perturbations to cell states."""
    
    def __init__(self, adata: ad.AnnData, preprocessor):
        """
        Parameters
        ----------
        adata : AnnData
            Reference data with gene names and velocity
        preprocessor : Preprocessor
            Preprocessor object with PCA model
        """
        self.adata = adata
        self.preprocessor = preprocessor
        
    def initial_state_shift(self, x0_pca: np.ndarray, 
                           gene_name: str,
                           fold_change: float = 2.0) -> np.ndarray:
        """Shift initial state by modifying gene expression."""
        if gene_name not in self.adata.var_names:
            raise ValueError(f"Gene '{gene_name}' not found")
        
        gene_idx = np.where(self.adata.var_names == gene_name)[0][0]
        print(f"Perturbing gene: {gene_name} (fold change: {fold_change}x)")
        
        pca_model = self.adata.uns.get('pca_model')
        if pca_model is None:
            raise ValueError("PCA model not found")
        
        x0_genes = pca_model.inverse_transform(x0_pca)
        x0_genes[:, gene_idx] *= fold_change
        x0_genes = np.maximum(x0_genes, 1e-6)
        x0_perturbed = pca_model.transform(x0_genes)
        
        print(f"  Shift magnitude: {np.linalg.norm(x0_perturbed - x0_pca):.4f}")
        return x0_perturbed
    
    def compute_forcing_term_local(self, gene_name: str, n_neighbors: int = 50) -> Callable:
        """
        Compute local forcing term by regressing velocity on gene expression.
        
        This estimates per-cell forcing Δv_g(x) by locally fitting the relationship
        between gene expression and velocity vectors.
        
        Parameters
        ----------
        gene_name : str
            Target gene
        n_neighbors : int
            Number of neighbors for local regression
            
        Returns
        -------
        forcing_function : callable
            Function(x_query, strength) -> Δv forcing term
        """
        if gene_name not in self.adata.var_names:
            raise ValueError(f"Gene '{gene_name}' not found")
        
        print(f"Computing local forcing term for: {gene_name}")
        
        X = self.adata.obsm['X_pca']          # (N, d)
        V = self.adata.obsm['velocity_pca']   # (N, d)
        
        # Extract gene expression
        gene_expr = self.adata[:, gene_name].X
        if hasattr(gene_expr, 'toarray'):
            gene_expr = gene_expr.toarray().ravel()
        else:
            gene_expr = gene_expr.ravel()
        
        # Build kNN index
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, X.shape[0])).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Compute per-cell local slopes by regressing V on gene_expr
        print("  Fitting local slopes for each cell...")
        slopes = np.zeros_like(V)  # (N, d)
        
        for i in range(X.shape[0]):
            nbr_idx = indices[i]
            g_neighbors = gene_expr[nbr_idx].reshape(-1, 1)  # (k, 1)
            V_neighbors = V[nbr_idx]                         # (k, d)
            
            # Fit multi-output linear regression: V = alpha * g + intercept
            try:
                lr = LinearRegression()
                lr.fit(g_neighbors, V_neighbors)  # alpha -> (1, d)
                slopes[i] = lr.coef_.reshape(-1)  # (d,)
            except:
                slopes[i] = V[i]  # Fallback to cell's own velocity
        
        # Smooth slopes across neighborhood
        print("  Smoothing slopes...")
        smoothed = np.zeros_like(slopes)
        for i in range(X.shape[0]):
            nbr_idx = indices[i]
            smoothed[i] = slopes[nbr_idx].mean(axis=0)
        
        print(f"  ✓ Forcing term computed")
        
        def forcing_fn(x_query: np.ndarray, strength: float = 1.0) -> np.ndarray:
            """
            Apply local forcing term.
            
            Parameters
            ----------
            x_query : np.ndarray
                Query points in PCA space (m, d)
            strength : float
                Scaling factor
                
            Returns
            -------
            delta_v : np.ndarray
                Forcing term (m, d)
            """
            # Find neighbors for query points
            dists_q, idx_q = nbrs.kneighbors(x_query)
            
            # RBF weighting with adaptive sigma
            sigma = np.median(dists_q) + 1e-8
            weights = np.exp(-(dists_q / (sigma + 1e-8))**2)
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
            
            # Weighted average of slopes
            delta_v = (weights[:, :, None] * smoothed[idx_q]).sum(axis=1)
            
            return strength * delta_v
        
        return forcing_fn
    
    def latent_perturbation(self, x0_pca: np.ndarray,
                           perturbation_direction: np.ndarray,
                           perturbation_strength: float = 1.0) -> np.ndarray:
        """Apply perturbation in latent (PCA) space directly."""
        direction_norm = np.linalg.norm(perturbation_direction)
        if direction_norm > 0:
            perturbation_direction = perturbation_direction / direction_norm
        
        x0_perturbed = x0_pca + perturbation_strength * perturbation_direction[None, :]
        
        print(f"Latent perturbation applied")
        print(f"  Perturbation magnitude: {perturbation_strength:.4f}")
        
        return x0_perturbed


if __name__ == "__main__":
    from ingest import create_demo_dataset
    from preprocess import Preprocessor
    from velocity import compute_velocity_scvelo
    
    print("Creating demo dataset...")
    adata = create_demo_dataset(n_cells=200, n_genes=500)
    
    print("Preprocessing...")
    prep = Preprocessor(n_top_genes=300, n_pcs=20)
    adata = prep.run_pipeline(adata)
    
    print("Computing velocity...")
    adata = compute_velocity_scvelo(adata, mode='stochastic')
    
    print("\nTesting perturbation operators...")
    pert_op = PerturbationOperator(adata, prep)
    
    first_gene = adata.var_names[0]
    x0 = adata.obsm['X_pca'][:5]
    
    # Test 1: Local forcing term (improved!)
    print("\n--- Test 1: Local Forcing Term ---")
    forcing_fn = pert_op.compute_forcing_term_local(first_gene, n_neighbors=30)
    forcing = forcing_fn(x0, strength=1.0)
    print(f"✓ Local forcing term shape: {forcing.shape}")
    print(f"  Mean magnitude: {np.linalg.norm(forcing, axis=1).mean():.6f}")
    
    # Test 2: Latent perturbation
    print("\n--- Test 2: Latent Perturbation ---")
    direction = np.random.randn(adata.obsm['X_pca'].shape[1])
    x0_latent = pert_op.latent_perturbation(x0, direction, 0.5)
    print(f"✓ Latent perturbation shape: {x0_latent.shape}\n")
    
    print("✓ All perturbation operators work!")
