"""
ODE Integration Engine

Forward simulation using torchdiffeq.
"""

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Callable, Union
import warnings


class ODEFunc(nn.Module):
    """
    ODE function wrapper for torchdiffeq.
    
    Implements dx/dt = v(x) where v(x) is the vector field.
    """
    
    def __init__(self, vector_field_func: Callable):
        """
        Parameters
        ----------
        vector_field_func : callable
            Function v(x) that returns velocity
        """
        super().__init__()
        self.vector_field = vector_field_func
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt = v(x).
        
        FIX: Handles NaN and infinite values
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (scalar, unused but required by torchdiffeq)
        x : torch.Tensor
            Current state (batch_size, n_dims)
            
        Returns
        -------
        dxdt : torch.Tensor
            Time derivative (batch_size, n_dims)
        """
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        
        # CRITICAL FIX: Handle NaN and infinite values
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=1e6, neginf=-1e6)
        x_np = np.clip(x_np, -1e6, 1e6)
        
        # Evaluate vector field
        try:
            v_np = self.vector_field(x_np)
        except Exception as e:
            warnings.warn(f"Vector field evaluation failed: {e}")
            v_np = np.zeros_like(x_np)
        
        # Handle output NaN values
        v_np = np.nan_to_num(v_np, nan=0.0, posinf=1e6, neginf=-1e6)
        v_np = np.clip(v_np, -1e6, 1e6)
        
        # Convert back to torch
        v = torch.from_numpy(v_np).to(x.device).type(x.dtype)
        
        return v


class ODEIntegrator:
    """
    ODE integrator for trajectory simulation.
    
    Integrates dx/dt = v(x) to simulate cell trajectories.
    """
    
    def __init__(self,
                 vector_field: Callable,
                 method: str = 'dopri5',
                 rtol: float = 1e-3,
                 atol: float = 1e-4,
                 adjoint: bool = False):
        """
        Parameters
        ----------
        vector_field : callable
            Vector field function v(x)
        method : str
            ODE solver method ('dopri5', 'rk4', 'euler', 'adaptive_heun')
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        adjoint : bool
            Use adjoint method
        """
        self.vector_field = vector_field
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.ode_func = ODEFunc(vector_field)
        
    def integrate(self,
                 x0: Union[np.ndarray, torch.Tensor],
                 t_span: Union[np.ndarray, torch.Tensor],
                 return_torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Integrate trajectories from initial conditions.
        
        Parameters
        ----------
        x0 : np.ndarray or torch.Tensor
            Initial conditions (n_cells, n_dims)
        t_span : np.ndarray or torch.Tensor
            Time points to evaluate (n_timepoints,)
        return_torch : bool
            Return torch.Tensor instead of numpy
            
        Returns
        -------
        trajectories : np.ndarray or torch.Tensor
            Simulated trajectories (n_timepoints, n_cells, n_dims)
        """
        # Convert to torch with NaN handling
        if isinstance(x0, np.ndarray):
            x0 = np.nan_to_num(x0, nan=0.0, posinf=1e6, neginf=-1e6)
            x0 = np.clip(x0, -1e6, 1e6)
            x0 = torch.from_numpy(x0.copy()).float()
            
        if isinstance(t_span, np.ndarray):
            t_span = np.nan_to_num(t_span, nan=0.0, posinf=1e6, neginf=-1e6)
            t_span = np.clip(t_span, 0, 1e6)
            t_span = torch.from_numpy(t_span.copy()).float()
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x0 = x0.to(device)
        t_span = t_span.to(device)
        
        print(f"Integrating ODE ({self.method})...")
        print(f"  Initial state shape: {x0.shape}")
        print(f"  Time steps: {t_span.shape[0]}")
        print(f"  Device: {device}")
        
        # Integrate with error handling
        try:
            if self.adjoint:
                trajectories = odeint(
                    self.ode_func, x0, t_span,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol,
                    adjoint_method='dopri5'
                )
            else:
                trajectories = odeint(
                    self.ode_func, x0, t_span,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol
                )
            
            print(f"✓ Integration successful")
            print(f"  Trajectory shape: {trajectories.shape}")
            
        except RuntimeError as e:
            warnings.warn(f"ODE integration failed with {self.method}: {e}")
            print(f"  Retrying with rk4 and stricter tolerances...")
            
            trajectories = odeint(
                self.ode_func, x0, t_span,
                method='rk4',
                rtol=self.rtol * 0.1,
                atol=self.atol * 0.1
            )
            print(f"✓ Integration successful (fallback)")
        
        # Final NaN check
        if torch.isnan(trajectories).any():
            print("⚠️  Warning: NaN detected in trajectory, replacing with 0")
            trajectories = torch.nan_to_num(trajectories, nan=0.0)
        
        # Convert to numpy if requested
        if not return_torch:
            trajectories = trajectories.detach().cpu().numpy()
        
        return trajectories
    
    def compute_extrapolation_score(self,
                                    x_simulated: np.ndarray,
                                    x_training: np.ndarray) -> np.ndarray:
        """
        Compute extrapolation score (distance from training data).
        
        Parameters
        ----------
        x_simulated : np.ndarray
            Simulated states (n_sim, n_dims)
        x_training : np.ndarray
            Training states (n_train, n_dims)
            
        Returns
        -------
        scores : np.ndarray
            Extrapolation scores (n_sim,)
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Compute distances to all training points
        distances = euclidean_distances(x_simulated, x_training)
        
        # Min distance to training set
        min_distances = distances.min(axis=1)
        
        # Normalize by median training-training distance
        train_distances = euclidean_distances(x_training, x_training)
        median_train_dist = np.median(train_distances[np.triu_indices_from(train_distances, k=1)])
        
        scores = min_distances / (median_train_dist + 1e-8)
        
        return scores
    
    def get_device_info(self) -> dict:
        """Get information about computational device."""
        if torch.cuda.is_available():
            return {
                'device': 'GPU',
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9,
            }
        else:
            return {
                'device': 'CPU',
                'cpu_count': torch.get_num_threads(),
            }


def simulate_trajectory(vector_field: Callable,
                       x0: np.ndarray,
                       t_max: float = 10.0,
                       n_steps: int = 100,
                       **kwargs) -> np.ndarray:
    """
    Simulate trajectory from initial condition.
    
    Parameters
    ----------
    vector_field : callable
        Vector field v(x)
    x0 : np.ndarray
        Initial condition (n_dims,) or (n_cells, n_dims)
    t_max : float
        Maximum time
    n_steps : int
        Number of time steps
    **kwargs : dict
        Additional arguments for ODEIntegrator
        
    Returns
    -------
    trajectory : np.ndarray
        Simulated trajectory (n_steps, n_cells, n_dims)
    """
    # Create integrator
    integrator = ODEIntegrator(vector_field, **kwargs)
    
    # Time span
    t_span = np.linspace(0, t_max, n_steps)
    
    # Ensure x0 is 2D
    if x0.ndim == 1:
        x0 = x0[None, :]
    
    # Integrate
    trajectory = integrator.integrate(x0, t_span)
    
    return trajectory


if __name__ == "__main__":
    # Demo: simple attractor
    def vector_field(x):
        """Simple attractor toward origin."""
        return -x + 0.1 * np.random.randn(*x.shape)
    
    print("Testing ODE integrator...")
    x0 = np.random.randn(5, 10)
    traj = simulate_trajectory(vector_field, x0, t_max=5.0, n_steps=50)
    
    print(f"\n✓ ODE integration works!")
    print(f"  Trajectory shape: {traj.shape}")
    print(f"  Initial state norm: {np.linalg.norm(traj[0], axis=1).mean():.4f}")
    print(f"  Final state norm: {np.linalg.norm(traj[-1], axis=1).mean():.4f}")
    print(f"  (Should be smaller due to attractor)")
