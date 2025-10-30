"""Tests for ODE integration engine."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from engine_explicit_field import ODEIntegrator, simulate_trajectory


class TestODEIntegrationStability:
    """Test ODE integration stability."""
    
    def test_simple_attractor_stability(self):
        """Test integration converges to attractor."""
        def simple_attractor(x):
            """Point toward origin."""
            return -x * 0.5
        
        x0 = np.random.randn(5, 10)
        integrator = ODEIntegrator(simple_attractor, method='dopri5')
        traj = integrator.integrate(x0, np.linspace(0, 10, 50))
        
        # Initial state should be far from origin
        init_dist = np.linalg.norm(traj[0])
        # Final state should be close to origin
        final_dist = np.linalg.norm(traj[-1])
        
        assert final_dist < init_dist, "Should converge to attractor"
    
    def test_stability_with_different_methods(self):
        """Test stability across different solvers."""
        def vf(x):
            return -x
        
        x0 = np.random.randn(3, 5)
        t_span = np.linspace(0, 1, 10)
        
        for method in ['dopri5', 'rk4']:
            integrator = ODEIntegrator(vf, method=method)
            traj = integrator.integrate(x0, t_span)
            
            # Check trajectory shape
            assert traj.shape == (10, 3, 5)
            # Check no NaN
            assert np.all(np.isfinite(traj))
    
    def test_integration_monotonicity(self):
        """Test that distance to origin decreases monotonically."""
        def attractor(x):
            return -x
        
        x0 = np.ones((3, 5))
        integrator = ODEIntegrator(attractor, method='rk4')
        traj = integrator.integrate(x0, np.linspace(0, 5, 20))
        
        # Compute distance from origin at each time step
        distances = np.linalg.norm(traj, axis=(1, 2))
        
        # Should be monotonically decreasing
        assert np.all(np.diff(distances) <= 0), "Distance should decrease"


class TestExtrapolationScore:
    """Test extrapolation score computation."""
    
    def test_extrapolation_in_manifold(self):
        """Test low score for points in manifold."""
        def vf(x):
            return -x
        
        integrator = ODEIntegrator(vf)
        
        # Generate training data in manifold
        X_train = np.random.randn(100, 5)
        
        # Points close to training data
        X_query = X_train[:10]
        
        # Compute extrapolation scores
        scores = integrator.compute_extrapolation_score(X_query, X_train)
        
        assert scores.shape == (10,)
        assert np.all(scores < 2.0), "In-manifold points should have low scores"
    
    def test_extrapolation_outside_manifold(self):
        """Test high score for points outside manifold."""
        def vf(x):
            return -x
        
        integrator = ODEIntegrator(vf)
        
        # Training data centered at origin
        X_train = np.random.randn(100, 5) * 0.1
        
        # Query far from training data
        X_query = np.ones((5, 5)) * 10.0
        
        # Compute scores
        scores = integrator.compute_extrapolation_score(X_query, X_train)
        
        assert np.all(scores > 1.0), "Out-of-manifold points should have high scores"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
