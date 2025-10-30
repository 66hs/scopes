"""
Velocity Module

RNA velocity computation using scVelo.
"""

import scanpy as sc
import scvelo as scv
import anndata as ad


def compute_velocity_scvelo(adata: ad.AnnData, 
                           mode: str = 'dynamical',
                           n_jobs: int = 1) -> ad.AnnData:
    """
    Compute RNA velocity using scVelo.
    
    Parameters
    ----------
    adata : AnnData
        Preprocessed data with spliced/unspliced layers
    mode : str
        'stochastic' or 'dynamical'
        - 'stochastic': Faster, simpler model
        - 'dynamical': More accurate, slower (requires fitting)
    n_jobs : int
        Number of parallel jobs (use 1 for Windows compatibility)
    
    Returns
    -------
    adata : AnnData
        Data with velocity estimates in obsm['velocity_pca'] and obsm['velocity_umap']
    """
    print(f"Computing velocity (scVelo {mode})...")
    
    # Check for required layers
    if 'spliced' not in adata.layers:
        print("  ⚠ No spliced layer found, creating from X")
        adata.layers['spliced'] = adata.X.copy()
        
    if 'unspliced' not in adata.layers:
        print("  ⚠ No unspliced layer found, creating simulated unspliced")
        # Simulate unspliced as 50% of spliced
        adata.layers['unspliced'] = adata.layers['spliced'].copy() * 0.5
    
    # Set verbosity
    scv.settings.verbosity = 3
    
    # Compute moments (preprocessing for velocity)
    print("  Computing moments...")
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    
    # Velocity estimation
    if mode == 'dynamical':
        print("  Recovering dynamics (this may take a few minutes)...")
        try:
            scv.tl.recover_dynamics(adata, n_jobs=n_jobs)
            scv.tl.velocity(adata, mode='dynamical')
            print("  ✓ Dynamical velocity computed")
        except Exception as e:
            print(f"  ⚠ Dynamical mode failed ({e}), falling back to stochastic")
            scv.tl.velocity(adata, mode='stochastic')
            print("  ✓ Stochastic velocity computed (fallback)")
    else:
        print("  Computing stochastic velocity...")
        scv.tl.velocity(adata, mode='stochastic')
        print("  ✓ Stochastic velocity computed")
    
    # Velocity graph
    print("  Computing velocity graph...")
    scv.tl.velocity_graph(adata)
    
    # Project velocity to embeddings
    print("  Projecting to PCA...")
    scv.tl.velocity_embedding(adata, basis='pca')
    
    if 'X_umap' in adata.obsm:
        print("  Projecting to UMAP...")
        scv.tl.velocity_embedding(adata, basis='umap')
    
    print("✓ Velocity computation complete")
    return adata


def compute_velocity_dynamo(adata: ad.AnnData) -> ad.AnnData:
    """
    Compute velocity using Dynamo (alternative to scVelo).
    
    Note: Dynamo has more advanced vector field reconstruction
    but is slower and more complex.
    
    Parameters
    ----------
    adata : AnnData
        Preprocessed data
    
    Returns
    -------
    adata : AnnData
        Data with Dynamo velocity field
    """
    try:
        import dynamo as dyn
        
        print("Computing velocity (Dynamo)...")
        
        # Dynamo preprocessing
        dyn.pp.recipe_monocle(adata)
        
        # Dynamics and velocity
        dyn.tl.dynamics(adata)
        dyn.tl.reduceDimension(adata)
        dyn.tl.cell_velocities(adata)
        
        # Vector field
        dyn.vf.VectorField(adata, basis='pca')
        
        print("✓ Dynamo velocity computed")
        return adata
        
    except ImportError:
        print("⚠ Dynamo not installed. Install with: pip install dynamo-release")
        print("  Falling back to scVelo...")
        return compute_velocity_scvelo(adata, mode='stochastic')


if __name__ == "__main__":
    # Demo
    from ingest import create_demo_dataset
    from preprocess import Preprocessor
    
    print("Creating demo dataset...")
    adata = create_demo_dataset(n_cells=500, n_genes=1000)
    
    print("\nPreprocessing...")
    preprocessor = Preprocessor(n_top_genes=500, n_pcs=30)
    adata = preprocessor.run_pipeline(adata)
    
    print("\nComputing velocity...")
    adata = compute_velocity_scvelo(adata, mode='stochastic')
    
    print(f"\n✓ Velocity shape: {adata.obsm['velocity_pca'].shape}")
    print(f"✓ Keys in obsm: {list(adata.obsm.keys())}")
