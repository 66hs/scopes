"""
Preprocessing Module

Normalization, feature selection, and dimensionality reduction.
"""

import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.decomposition import PCA


class Preprocessor:
    """Handles preprocessing pipeline for single-cell data."""
    
    def __init__(self,
                 n_top_genes: int = 2000,
                 n_pcs: int = 50,
                 n_neighbors: int = 20):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        n_top_genes : int
            Number of highly variable genes to select
        n_pcs : int
            Number of principal components
        n_neighbors : int
            Number of neighbors for kNN graph
        """
        self.n_top_genes = n_top_genes
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.pca = None
        self.scaler = None
        
    def run_pipeline(self, adata: ad.AnnData,
                    skip_highly_variable: bool = False) -> ad.AnnData:
        """
        Run complete preprocessing pipeline.
        
        Parameters
        ----------
        adata : AnnData
            Input data
        skip_highly_variable : bool
            Skip HVG selection (use all genes)
            
        Returns
        -------
        adata : AnnData
            Preprocessed data with PCA, neighbors, UMAP
        """
        print("Running preprocessing pipeline...")
        
        # QC filtering
        adata = self.qc_filter(adata)
        
        # Normalization
        adata = self.normalize(adata)
        
        # Feature selection
        if not skip_highly_variable:
            adata = self.select_highly_variable_genes(adata)
        
        # Dimensionality reduction
        adata = self.compute_pca(adata)
        
        # Neighborhood graph
        adata = self.compute_neighbors(adata)
        
        # UMAP
        adata = self.compute_umap(adata)
        
        print("✓ Preprocessing complete")
        return adata
    
    def qc_filter(self, adata: ad.AnnData,
                 min_genes: int = 200,
                 min_cells: int = 3,
                 max_pct_mito: float = 20.0) -> ad.AnnData:
        """Filter cells and genes based on QC metrics."""
        print("  QC filtering...")
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        # Filter cells
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
        # Filter genes
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        print(f"    Cells: {n_cells_before} → {adata.n_obs}")
        print(f"    Genes: {n_genes_before} → {adata.n_vars}")
        
        return adata
    
    def normalize(self, adata: ad.AnnData,
                 target_sum: float = 1e4) -> ad.AnnData:
        """Normalize and log-transform."""
        print("  Normalizing...")
        
        # Store raw counts
        adata.layers['counts'] = adata.X.copy()
        
        # Normalize to target sum
        sc.pp.normalize_total(adata, target_sum=target_sum)
        
        # Log transform
        sc.pp.log1p(adata)
        
        # Store normalized
        adata.layers['normalized'] = adata.X.copy()
        
        return adata
    
    def select_highly_variable_genes(self, adata: ad.AnnData) -> ad.AnnData:
        """Select highly variable genes."""
        print(f"  Selecting {self.n_top_genes} HVGs...")
        
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=self.n_top_genes,
            flavor='seurat_v3',
            subset=True
        )
        
        print(f"    ✓ Selected {adata.n_vars} highly variable genes")
        return adata
    
    def compute_pca(self, adata: ad.AnnData) -> ad.AnnData:
        """Compute PCA."""
        print(f"  Computing PCA ({self.n_pcs} PCs)...")
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
        
        # PCA
        sc.tl.pca(adata, n_comps=self.n_pcs, svd_solver='arpack')
        
        # Store PCA model for later use
        self.pca = PCA(n_components=self.n_pcs)
        self.pca.fit(adata.X)
        
        # Store in adata for later access
        adata.uns['pca_model'] = self.pca
        
        variance_explained = adata.uns['pca']['variance_ratio'].sum()
        print(f"    ✓ Explained variance: {variance_explained:.2%}")
        
        return adata
    
    def compute_neighbors(self, adata: ad.AnnData) -> ad.AnnData:
        """Compute neighborhood graph."""
        print(f"  Computing neighbors (k={self.n_neighbors})...")
        
        sc.pp.neighbors(
            adata,
            n_neighbors=self.n_neighbors,
            n_pcs=self.n_pcs,
            method='umap'
        )
        
        return adata
    
    def compute_umap(self, adata: ad.AnnData) -> ad.AnnData:
        """Compute UMAP embedding."""
        print("  Computing UMAP...")
        
        sc.tl.umap(adata)
        
        print("    ✓ UMAP computed")
        return adata
    
    def inverse_transform_pca(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Transform from PCA space back to gene space.
        
        Parameters
        ----------
        X_pca : np.ndarray
            Data in PCA space (n_cells, n_pcs)
            
        Returns
        -------
        X_genes : np.ndarray
            Data in gene space (n_cells, n_genes)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Run compute_pca first.")
        
        return self.pca.inverse_transform(X_pca)


if __name__ == "__main__":
    # Demo
    from ingest import create_demo_dataset
    
    adata = create_demo_dataset()
    preprocessor = Preprocessor(n_top_genes=500, n_pcs=30)
    adata_proc = preprocessor.run_pipeline(adata)
    print(adata_proc)
