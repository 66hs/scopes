"""
Simulate Module

High-level API for perturbation simulation.
"""

import numpy as np
import anndata as ad
from typing import Optional, Dict, Callable, Tuple
from pathlib import Path

from ingest import load_h5ad
from preprocess import Preprocessor
from velocity import compute_velocity_scvelo
from vectorfield import fit_vector_field_from_scvelo, VectorFieldInterpolator
from engine_explicit_field import ODEIntegrator
from perturbation import PerturbationOperator
from decoder import train_decoder, decode_trajectory


class PerturbationSimulator:
    """
    Main API for simulating cell state perturbations.
    
    Workflow:
    1. Load data
    2. Preprocess
    3. Compute velocity
    4. Fit vector field
    5. Train decoder
    6. Simulate with perturbations
    """
    
    def __init__(self, adata: ad.AnnData = None,
                 data_path: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize simulator.
        
        Parameters
        ----------
        adata : AnnData, optional
            Preprocessed data object
        data_path : str, optional
            Path to .h5ad file to load
        verbose : bool
            Print progress messages
        """
        self.verbose = verbose
        self.adata = None
        self.preprocessor = None
        self.vector_field = None
        self.integrator = None
        self.decoder = None
        self.perturbation_op = None
        
        if adata is not None:
            self.adata = adata
        elif data_path is not None:
            self._load_data(data_path)
        
    def _load_data(self, data_path: str):
        """Load data from .h5ad file."""
        if self.verbose:
            print(f"Loading data from {data_path}...")
        self.adata = load_h5ad(data_path, validate=True)
        if self.verbose:
            print(f"✓ Loaded {self.adata.n_obs} cells × {self.adata.n_vars} genes")
    
    def preprocess(self, n_top_genes: int = 2000, n_pcs: int = 50,
                  n_neighbors: int = 20, **kwargs):
        """
        Run preprocessing pipeline.
        
        Parameters
        ----------
        n_top_genes : int
            Highly variable genes to select
        n_pcs : int
            Number of PCA components
        n_neighbors : int
            Number of neighbors for kNN graph
        """
        if self.verbose:
            print("Running preprocessing...")
        
        self.preprocessor = Preprocessor(
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors
        )
        self.adata = self.preprocessor.run_pipeline(self.adata)
        
        if self.verbose:
            print(f"✓ Preprocessing complete")
    
    def compute_velocity(self, mode: str = 'stochastic'):
        """
        Compute RNA velocity.
        
        Parameters
        ----------
        mode : str
            'stochastic' or 'dynamical'
        """
        if self.verbose:
            print("Computing velocity...")
        
        self.adata = compute_velocity_scvelo(self.adata, mode=mode)
        
        if self.verbose:
            print(f"✓ Velocity computed")
    
    def fit_vector_field(self, method: str = 'knn_rbf', n_neighbors: int = 30):
        """
        Fit vector field from velocity estimates.
        
        Parameters
        ----------
        method : str
            'knn_rbf' or 'mlp'
        n_neighbors : int
            Number of neighbors
        """
        if self.verbose:
            print("Fitting vector field...")
        
        self.vector_field = fit_vector_field_from_scvelo(
            self.adata,
            method=method,
            basis='pca',
            n_neighbors=n_neighbors
        )
        
        if self.verbose:
            print(f"✓ Vector field fitted")
    
    def train_decoder(self, n_epochs: int = 50, batch_size: int = 64):
        """
        Train expression decoder.
        
        Parameters
        ----------
        n_epochs : int
            Training epochs
        batch_size : int
            Batch size
        """
        if self.verbose:
            print("Training decoder...")
        
        self.decoder = train_decoder(
            self.adata,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"✓ Decoder trained")
    
    def setup_perturbation_operator(self):
        """Setup perturbation operator."""
        if self.preprocessor is None:
            raise ValueError("Run preprocess() first")
        
        self.perturbation_op = PerturbationOperator(self.adata, self.preprocessor)
    
    def simulate_trajectory(self,
                           x0_pca: np.ndarray,
                           t_max: float = 10.0,
                           n_steps: int = 100,
                           return_genes: bool = True) -> Dict:
        """
        Simulate cell trajectory from initial state.
        
        Parameters
        ----------
        x0_pca : np.ndarray
            Initial state in PCA space (n_cells, n_pcs)
        t_max : float
            Maximum simulation time
        n_steps : int
            Number of time steps
        return_genes : bool
            Decode trajectory to gene space
            
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'trajectory_pca': (n_steps, n_cells, n_pcs)
            - 'trajectory_genes': (n_steps, n_cells, n_genes) if return_genes
            - 't_span': time points
        """
        if self.vector_field is None:
            raise ValueError("Run fit_vector_field() first")
        
        if self.verbose:
            print(f"Simulating trajectory for {x0_pca.shape[0]} cells...")
        
        # Create integrator
        self.integrator = ODEIntegrator(self.vector_field)
        
        # Time span
        t_span = np.linspace(0, t_max, n_steps)
        
        # Integrate
        trajectory_pca = self.integrator.integrate(x0_pca, t_span, return_torch=False)
        
        result = {
            'trajectory_pca': trajectory_pca,
            't_span': t_span
        }
        
        # Decode to gene space if decoder available
        if return_genes and self.decoder is not None:
            if self.verbose:
                print("Decoding trajectory to gene space...")
            result['trajectory_genes'] = decode_trajectory(self.decoder, trajectory_pca)
        
        if self.verbose:
            print(f"✓ Simulation complete")
        
        return result
    
    def simulate_perturbation(self,
                             initial_cells: np.ndarray,
                             perturbation_gene: str,
                             fold_change: float = 2.0,
                             t_max: float = 10.0,
                             n_steps: int = 100) -> Dict:
        """
        Simulate trajectory under gene perturbation.
        
        Parameters
        ----------
        initial_cells : np.ndarray
            Initial cell states in PCA space (n_cells, n_pcs)
        perturbation_gene : str
            Gene to perturb
        fold_change : float
            Fold change (2.0 = 2x upregulation)
        t_max : float
            Simulation time
        n_steps : int
            Number of time steps
            
        Returns
        -------
        result : dict
            Simulation results including perturbed and wild-type trajectories
        """
        if self.perturbation_op is None:
            self.setup_perturbation_operator()
        
        if self.verbose:
            print(f"Simulating perturbation: {perturbation_gene} (fold change: {fold_change}x)")
        
        # Wild-type trajectory
        wt_result = self.simulate_trajectory(initial_cells, t_max=t_max, n_steps=n_steps)
        
        # Perturbed initial state
        x0_perturbed = self.perturbation_op.initial_state_shift(
            initial_cells, perturbation_gene, fold_change=fold_change
        )
        
        # Perturbed trajectory
        perturbed_result = self.simulate_trajectory(x0_perturbed, t_max=t_max, n_steps=n_steps)
        
        result = {
            'wt_trajectory_pca': wt_result['trajectory_pca'],
            'perturbed_trajectory_pca': perturbed_result['trajectory_pca'],
            't_span': wt_result['t_span'],
            'perturbation_gene': perturbation_gene,
            'fold_change': fold_change
        }
        
        if 'trajectory_genes' in wt_result:
            result['wt_trajectory_genes'] = wt_result['trajectory_genes']
            result['perturbed_trajectory_genes'] = perturbed_result['trajectory_genes']
        
        if self.verbose:
            print(f"✓ Perturbation simulation complete")
        
        return result
    
    def compute_effect_size(self, wt_trajectory: np.ndarray,
                           perturbed_trajectory: np.ndarray) -> np.ndarray:
        """
        Compute effect size (L2 distance) between trajectories over time.
        
        Parameters
        ----------
        wt_trajectory : np.ndarray
            Wild-type trajectory (n_steps, n_cells, n_dims)
        perturbed_trajectory : np.ndarray
            Perturbed trajectory (n_steps, n_cells, n_dims)
            
        Returns
        -------
        effect_size : np.ndarray
            Effect size per time step (n_steps,)
        """
        distances = np.linalg.norm(
            perturbed_trajectory - wt_trajectory, axis=(1, 2)
        )
        return distances


if __name__ == "__main__":
    # Demo: Full simulation pipeline
    from ingest import create_demo_dataset
        
    print("=" * 60)
    print("PERTURBATION SIMULATOR DEMO")
    print("=" * 60)
    
    # Create simulator
    adata = create_demo_dataset(n_cells=200, n_genes=500)
    sim = PerturbationSimulator(adata=adata, verbose=True)
    
    # Preprocessing
    print("\n1. PREPROCESSING")
    sim.preprocess(n_top_genes=300, n_pcs=20, n_neighbors=20)
    
    # Velocity
    print("\n2. VELOCITY COMPUTATION")
    sim.compute_velocity(mode='stochastic')
    
    # Vector field
    print("\n3. VECTOR FIELD FITTING")
    sim.fit_vector_field(method='knn_rbf', n_neighbors=30)
    
    # Decoder
    print("\n4. DECODER TRAINING")
    sim.train_decoder(n_epochs=30, batch_size=32)
    
    # Simulation
    print("\n5. TRAJECTORY SIMULATION")
    x0 = sim.adata.obsm['X_pca'][:3]  # First 3 cells
    result = sim.simulate_trajectory(x0, t_max=5, n_steps=20, return_genes=True)
    print(f"  WT trajectory shape: {result['trajectory_pca'].shape}")
    print(f"  Decoded to genes: {result['trajectory_genes'].shape}")
    
    print("\n" + "=" * 60)
    print("✓ SIMULATOR DEMO COMPLETE")
    print("=" * 60)
