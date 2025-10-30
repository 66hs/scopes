"""Tests for vector field interpolation."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vectorfield import VectorFieldInterpolator


class TestKNNRBFInterpolation:
    """Test kNN+RBF interpolation."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.V_train = -self.X_train + 0.1 * np.random.randn(100, 10)
    
    def test_knn_rbf_initialization(self):
        """Test kNN+RBF initialization."""
        vf = VectorFieldInterpolator(method='knn_rbf', n_neighbors=30)
        assert vf.method == 'knn_rbf'
    
    def test_knn_rbf_fitting(self):
        """Test kNN+RBF fitting."""
        vf = VectorFieldInterpolator(method='knn_rbf')
        vf.fit(self.X_train, self.V_train)
        assert vf.knn is not None
    
    def test_knn_rbf_prediction(self):
        """Test kNN+RBF prediction."""
        vf = VectorFieldInterpolator(method='knn_rbf')
        vf.fit(self.X_train, self.V_train)
        
        X_query = np.random.randn(10, 10)
        V_pred = vf(X_query)
        
        assert V_pred.shape == (10, 10)
        assert np.all(np.isfinite(V_pred))
    
    def test_knn_rbf_prediction_accuracy(self):
        """Test kNN+RBF prediction on training data."""
        vf = VectorFieldInterpolator(method='knn_rbf', n_neighbors=10)
        vf.fit(self.X_train, self.V_train)
        
        # Predict on subset of training data
        V_pred = vf(self.X_train[:20])
        
        # Should have reasonable prediction
        mse = np.mean((V_pred - self.V_train[:20]) ** 2)
        assert mse < 1.0  # Reasonable error


class TestVectorFieldGeneral:
    """Test general vector field functionality."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X_train = np.random.randn(80, 8)
        self.V_train = -self.X_train + 0.05 * np.random.randn(80, 8)
    
    def test_vf_initialization(self):
        """Test vector field initialization."""
        vf = VectorFieldInterpolator(method='knn_rbf')
        assert vf is not None
    
    def test_vf_fitting_and_calling(self):
        """Test fitting and calling vector field."""
        vf = VectorFieldInterpolator(method='knn_rbf', n_neighbors=20)
        vf.fit(self.X_train, self.V_train)
        
        # Should be callable
        X_query = np.random.randn(5, 8)
        V_query = vf(X_query)
        
        assert V_query.shape == (5, 8)
    
    def test_vf_output_shapes(self):
        """Test various query shapes."""
        vf = VectorFieldInterpolator(method='knn_rbf')
        vf.fit(self.X_train, self.V_train)
        
        for n_query in [1, 5, 10, 50]:
            X_query = np.random.randn(n_query, 8)
            V_query = vf(X_query)
            assert V_query.shape == (n_query, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
