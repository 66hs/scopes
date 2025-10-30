"""
Validation Module

Metrics and validation functions for simulation accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import anndata as ad


class SimulationValidator:
    """Validate simulation quality and accuracy."""
    
    def __init__(self, adata_original: ad.AnnData):
        """
        Parameters
        ----------
        adata_original : AnnData
            Original reference data for validation
        """
        self.adata = adata_original
        
    def compute_trajectory_smoothness(self, trajectory: np.ndarray) -> float:
        """
        Compute trajectory smoothness (lower = smoother).
        
        Measures second derivative of trajectory.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Trajectory (n_steps, n_cells, n_dims)
            
        Returns
        -------
        smoothness : float
            Mean squared acceleration across trajectory
        """
        # Compute first derivative (velocity)
        v1 = np.diff(trajectory, axis=0)
        
        # Compute second derivative (acceleration)
        a = np.diff(v1, axis=0)
        
        # Compute smoothness as mean squared acceleration
        smoothness = np.mean(np.linalg.norm(a, axis=2) ** 2)
        
        return smoothness
    
    def compute_manifold_consistency(self, 
                                     trajectory_pca: np.ndarray,
                                     trajectory_genes: np.ndarray) -> float:
        """
        Compute consistency between PCA and gene space trajectories.
        
        Compares distance profiles in both spaces.
        
        Parameters
        ----------
        trajectory_pca : np.ndarray
            Trajectory in PCA space (n_steps, n_cells, n_pcs)
        trajectory_genes : np.ndarray
            Trajectory in gene space (n_steps, n_cells, n_genes)
            
        Returns
        -------
        consistency : float
            Correlation between distance profiles (0-1, higher = better)
        """
        # Compute pairwise distances in PCA space
        n_steps = trajectory_pca.shape[0]
        dist_pca = np.zeros((n_steps - 1,))
        
        for t in range(n_steps - 1):
            dist_pca[t] = np.linalg.norm(trajectory_pca[t+1] - trajectory_pca[t])
        
        # Compute pairwise distances in gene space
        dist_genes = np.zeros((n_steps - 1,))
        
        for t in range(n_steps - 1):
            dist_genes[t] = np.linalg.norm(trajectory_genes[t+1] - trajectory_genes[t])
        
        # Normalize distances
        dist_pca = dist_pca / (np.max(dist_pca) + 1e-8)
        dist_genes = dist_genes / (np.max(dist_genes) + 1e-8)
        
        # Compute correlation between distance profiles
        consistency = np.corrcoef(dist_pca, dist_genes)[0, 1]
        
        # Handle NaN
        if np.isnan(consistency):
            consistency = 0.0
        
        return consistency
    
    def compute_perturbation_effect_size(self,
                                         wt_trajectory: np.ndarray,
                                         pert_trajectory: np.ndarray) -> Dict:
        """
        Compute effect size metrics for perturbation.
        
        Parameters
        ----------
        wt_trajectory : np.ndarray
            Wild-type trajectory (n_steps, n_cells, n_dims)
        pert_trajectory : np.ndarray
            Perturbed trajectory (n_steps, n_cells, n_dims)
            
        Returns
        -------
        metrics : dict
            Dictionary with effect size metrics
        """
        # L2 distance over time
        distances = np.linalg.norm(pert_trajectory - wt_trajectory, axis=(1, 2))
        
        metrics = {
            'mean_effect_size': np.mean(distances),
            'max_effect_size': np.max(distances),
            'min_effect_size': np.min(distances),
            'effect_size_std': np.std(distances),
            'early_effect': np.mean(distances[:len(distances)//3]),  # First third
            'late_effect': np.mean(distances[2*len(distances)//3:]),  # Last third
        }
        
        return metrics
    
    def compute_velocity_consistency(self,
                                      trajectory: np.ndarray,
                                      vector_field_fn) -> float:
        """
        Check if simulated trajectory is consistent with vector field.
        
        Computes correlation between numerical derivatives and vector field predictions.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Trajectory (n_steps, n_cells, n_dims)
        vector_field_fn : callable
            Vector field function
            
        Returns
        -------
        consistency : float
            Velocity consistency (0-1, higher = better)
        """
        # Compute numerical derivatives
        dt = 1.0  # Assume unit time steps
        v_numerical = np.diff(trajectory, axis=0) / dt
        
        # Get vector field predictions at midpoints
        trajectory_mid = (trajectory[:-1] + trajectory[1:]) / 2
        trajectory_flat = trajectory_mid.reshape(-1, trajectory_mid.shape[2])
        
        v_predicted = vector_field_fn(trajectory_flat)
        v_predicted = v_predicted.reshape(trajectory_mid.shape)
        
        # Compute correlation of magnitudes
        v_numerical_mag = np.linalg.norm(v_numerical, axis=2)
        v_predicted_mag = np.linalg.norm(v_predicted, axis=2)
        
        v_numerical_flat = v_numerical_mag.flatten()
        v_predicted_flat = v_predicted_mag.flatten()
        
        consistency = np.corrcoef(v_numerical_flat, v_predicted_flat)[0, 1]
        
        if np.isnan(consistency):
            consistency = 0.0
        
        return consistency
    
    def compute_reconstruction_error(self,
                                     decoder,
                                     trajectory_pca: np.ndarray,
                                     reference_genes: np.ndarray) -> float:
        """
        Compute reconstruction error for decoded trajectory.
        
        Parameters
        ----------
        decoder : ExpressionDecoder
            Trained decoder model
        trajectory_pca : np.ndarray
            Trajectory in PCA space
        reference_genes : np.ndarray
            Reference gene expression
            
        Returns
        -------
        rmse : float
            Root mean squared error
        """
        import torch
        
        device = next(decoder.parameters()).device
        decoder.eval()
        
        # Flatten trajectory
        traj_flat = trajectory_pca.reshape(-1, trajectory_pca.shape[-1])
        traj_tensor = torch.FloatTensor(traj_flat).to(device)
        
        with torch.no_grad():
            expr_pred = decoder(traj_tensor).cpu().numpy()
        
        # Reshape back
        expr_pred = expr_pred.reshape(trajectory_pca.shape[0], trajectory_pca.shape[1], -1)
        
        # Compute RMSE
        mse = np.mean((expr_pred - reference_genes) ** 2)
        rmse = np.sqrt(mse)
        
        return rmse
    
    def compute_biological_validity(self, trajectory_genes: np.ndarray) -> Dict:
        """
        Check biological validity of gene expression trajectory.
        
        Parameters
        ----------
        trajectory_genes : np.ndarray
            Gene expression trajectory (n_steps, n_cells, n_genes)
            
        Returns
        -------
        metrics : dict
            Biological validity metrics
        """
        metrics = {}
        
        # 1. Gene expression should be non-negative
        n_negative = np.sum(trajectory_genes < 0)
        metrics['negative_expression_fraction'] = n_negative / trajectory_genes.size
        
        # 2. Gene expression should be finite
        n_nan = np.sum(~np.isfinite(trajectory_genes))
        metrics['nan_fraction'] = n_nan / trajectory_genes.size
        
        # 3. Expression should be within reference range
        ref_genes = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        ref_min, ref_max = ref_genes.min(), ref_genes.max()
        
        n_out_of_range = np.sum((trajectory_genes < ref_min) | (trajectory_genes > ref_max))
        metrics['out_of_range_fraction'] = n_out_of_range / trajectory_genes.size
        
        # 4. Compute validity score (0-1, higher = better)
        validity_score = 1.0 - (metrics['negative_expression_fraction'] + 
                                metrics['nan_fraction'] + 
                                metrics['out_of_range_fraction']) / 3
        metrics['validity_score'] = validity_score
        
        return metrics


class MetricsReporter:
    """Generate validation reports."""
    
    def __init__(self, validator: SimulationValidator):
        """
        Parameters
        ----------
        validator : SimulationValidator
            Validator instance
        """
        self.validator = validator
    
    def generate_report(self, 
                       trajectory_pca: np.ndarray,
                       trajectory_genes: np.ndarray,
                       vector_field_fn,
                       decoder) -> pd.DataFrame:
        """
        Generate comprehensive validation report.
        
        Parameters
        ----------
        trajectory_pca : np.ndarray
            Trajectory in PCA space
        trajectory_genes : np.ndarray
            Trajectory in gene space
        vector_field_fn : callable
            Vector field function
        decoder : ExpressionDecoder
            Trained decoder
            
        Returns
        -------
        report : pd.DataFrame
            Metrics report
        """
        print("Computing validation metrics...")
        
        metrics = {}
        
        # 1. Trajectory smoothness
        smoothness = self.validator.compute_trajectory_smoothness(trajectory_pca)
        metrics['Trajectory_Smoothness'] = smoothness
        print(f"  Smoothness: {smoothness:.6f}")
        
        # 2. Manifold consistency
        consistency = self.validator.compute_manifold_consistency(trajectory_pca, trajectory_genes)
        metrics['Manifold_Consistency'] = consistency
        print(f"  Manifold Consistency: {consistency:.6f}")
        
        # 3. Velocity consistency
        vel_consistency = self.validator.compute_velocity_consistency(trajectory_pca, vector_field_fn)
        metrics['Velocity_Consistency'] = vel_consistency
        print(f"  Velocity Consistency: {vel_consistency:.6f}")
        
        # 4. Reconstruction error
        rmse = self.validator.compute_reconstruction_error(decoder, trajectory_pca, trajectory_genes)
        metrics['Reconstruction_RMSE'] = rmse
        print(f"  Reconstruction RMSE: {rmse:.6f}")
        
        # 5. Biological validity
        bio_validity = self.validator.compute_biological_validity(trajectory_genes)
        metrics.update(bio_validity)
        print(f"  Biological Validity Score: {bio_validity['validity_score']:.6f}")
        
        # Create report
        report = pd.DataFrame([metrics]).T
        report.columns = ['Value']
        
        return report


if __name__ == "__main__":
    # Demo
    from ingest import create_demo_dataset
    from preprocess import Preprocessor
    from velocity import compute_velocity_scvelo
    from vectorfield import fit_vector_field_from_scvelo
    from decoder import train_decoder, decode_trajectory
    from engine_explicit_field import simulate_trajectory
    
    print("Setting up validation demo...")
    adata = create_demo_dataset(n_cells=200, n_genes=500)
    
    print("Preprocessing...")
    prep = Preprocessor(n_top_genes=300, n_pcs=20)
    adata = prep.run_pipeline(adata)
    
    print("Computing velocity...")
    adata = compute_velocity_scvelo(adata, mode='stochastic')
    
    print("Fitting vector field...")
    vf = fit_vector_field_from_scvelo(adata, method='knn_rbf', basis='pca')
    
    print("Training decoder...")
    decoder = train_decoder(adata, n_epochs=20, verbose=False)
    
    # Create validator
    validator = SimulationValidator(adata)
    reporter = MetricsReporter(validator)
    
    # Simulate trajectory
    print("Simulating trajectory...")
    x0 = adata.obsm['X_pca'][:3]
    traj_pca = simulate_trajectory(vf, x0, t_max=5, n_steps=20)
    
    # Decode
    print("Decoding trajectory...")
    traj_genes = decode_trajectory(decoder, traj_pca)
    
    # Generate report
    print("\nGenerating validation report...")
    report = reporter.generate_report(traj_pca, traj_genes, vf, decoder)
    
    print("\n" + "="*60)
    print("VALIDATION METRICS REPORT")
    print("="*60)
    print(report)
    print("="*60)
    
    print("\n✓ Validation module works!")
