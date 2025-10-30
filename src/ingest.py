"""
Data Ingestion Module

Handles loading and validation of .h5ad files for perturbation simulation.
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings


class DataValidator:
    """Validates input data for perturbation simulation."""
    
    def __init__(self, min_genes: int = 200, min_cells: int = 3):
        self.min_genes = min_genes
        self.min_cells = min_cells
        
    def validate(self, adata: ad.AnnData) -> Tuple[bool, list]:
        """
        Validate AnnData object for simulation requirements.
        
        Parameters
        ----------
        adata : AnnData
            Input data object
            
        Returns
        -------
        valid : bool
            Whether data passes validation
        warnings : list
            List of warning messages
        """
        issues = []
        
        # Check dimensions
        n_cells, n_genes = adata.shape
        if n_cells < self.min_cells:
            issues.append(f"❌ Too few cells: {n_cells} < {self.min_cells}")
            return False, issues
        if n_genes < self.min_genes:
            issues.append(f"❌ Too few genes: {n_genes} < {self.min_genes}")
            return False, issues
            
        # Check for required layers
        if 'spliced' in adata.layers and 'unspliced' in adata.layers:
            issues.append("✓ Found spliced/unspliced layers")
        else:
            issues.append("⚠ Missing spliced/unspliced layers (will use total counts)")
            
        # Check for sparse format
        if not hasattr(adata.X, 'data'):  # Check if sparse
            issues.append("⚠ Data not sparse (memory usage may be high)")
            
        # Check for NaN or infinite values
        X_dense = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        if np.any(np.isnan(X_dense)):
            issues.append("❌ NaN values detected in expression matrix")
            return False, issues
            
        return True, issues


def load_h5ad(filepath: Union[str, Path],
              backed: bool = False,
              validate: bool = True) -> ad.AnnData:
    """
    Load .h5ad file with validation.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .h5ad file
    backed : bool, default False
        Whether to load in backed mode (for large files)
    validate : bool, default True
        Whether to validate data
        
    Returns
    -------
    adata : AnnData
        Loaded and validated data object
        
    Examples
    --------
    >>> adata = load_h5ad('data/pbmc3k.h5ad')
    >>> print(adata)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
        
    if filepath.suffix != '.h5ad':
        raise ValueError(f"Expected .h5ad file, got {filepath.suffix}")
    
    # Load data
    print(f"Loading {filepath.name}...")
    if backed:
        adata = sc.read_h5ad(filepath, backed='r')
    else:
        adata = sc.read_h5ad(filepath)
    
    print(f"✓ Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Validate
    if validate:
        validator = DataValidator()
        valid, issues = validator.validate(adata)
        
        for issue in issues:
            if issue.startswith('❌'):
                raise ValueError(issue)
            elif issue.startswith('⚠'):
                warnings.warn(issue)
            else:
                print(issue)
                
        if not valid:
            raise ValueError("Data validation failed")
    
    return adata


def create_demo_dataset(n_cells: int = 2000, 
                       n_genes: int = 1000,
                       n_hvg: int = 500,
                       save_path: Optional[str] = None) -> ad.AnnData:
    """
    Create synthetic demo dataset for testing.
    
    Parameters
    ----------
    n_cells : int
        Number of cells to generate
    n_genes : int
        Number of genes to generate
    n_hvg : int
        Number of highly variable genes
    save_path : str, optional
        Path to save the demo dataset
        
    Returns
    -------
    adata : AnnData
        Demo dataset with simulated trajectory
    """
    np.random.seed(42)
    
    # Simulate developmental trajectory
    pseudotime = np.linspace(0, 1, n_cells)
    
    # Generate expression data
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Add trajectory structure
    for i in range(n_hvg):
        # Some genes increase
        if i < n_hvg // 3:
            X[:, i] = X[:, i] * (1 + 3 * pseudotime)
        # Some genes decrease  
        elif i < 2 * n_hvg // 3:
            X[:, i] = X[:, i] * (1 + 3 * (1 - pseudotime))
        # Some genes transient
        else:
            X[:, i] = X[:, i] * (1 + 5 * np.sin(pseudotime * np.pi))
    
    # Create AnnData
    adata = ad.AnnData(X)
    adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
    adata.var_names = [f'Gene_{i}' for i in range(n_genes)]
    
    # Add metadata
    adata.obs['pseudotime'] = pseudotime
    adata.obs['cluster'] = pd.cut(pseudotime, bins=3, labels=['Early', 'Mid', 'Late'])
    
    # Add velocity layers (simulated)
    adata.layers['spliced'] = X.copy()
    adata.layers['unspliced'] = X.copy() * np.random.uniform(0.5, 1.5, size=X.shape)
    
    print(f"✓ Created demo dataset: {n_cells} cells × {n_genes} genes")
    
    if save_path:
        adata.write_h5ad(save_path)
        print(f"✓ Saved to {save_path}")
    
    return adata


if __name__ == "__main__":
    # Demo usage
    demo = create_demo_dataset(save_path='data/demo.h5ad')
    print(demo)
