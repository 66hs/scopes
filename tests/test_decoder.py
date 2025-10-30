"""Tests for expression decoder."""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from decoder import ExpressionDecoder


class TestExpressionDecoderNetwork:
    """Test expression decoder neural network."""
    
    def test_decoder_initialization(self):
        """Test decoder creation."""
        decoder = ExpressionDecoder(n_pcs=20, n_genes=300)
        assert decoder.n_pcs == 20
        assert decoder.n_genes == 300
    
    def test_decoder_forward_pass(self):
        """Test forward pass."""
        decoder = ExpressionDecoder(n_pcs=15, n_genes=200)
        X_pca = torch.randn(10, 15)
        
        with torch.no_grad():
            X_pred = decoder(X_pca)
        
        assert X_pred.shape == (10, 200)
        assert torch.all(torch.isfinite(X_pred))
    
    def test_decoder_batch_processing(self):
        """Test decoder with different batch sizes."""
        decoder = ExpressionDecoder(n_pcs=20, n_genes=300)
        
        for batch_size in [1, 5, 32, 100]:
            X_pca = torch.randn(batch_size, 20)
            X_pred = decoder(X_pca)
            assert X_pred.shape == (batch_size, 300)
    
    def test_decoder_output_finiteness(self):
        """Test decoder output is finite."""
        decoder = ExpressionDecoder(n_pcs=15, n_genes=150)
        X_pca = torch.randn(20, 15)
        
        X_pred = decoder(X_pca)
        
        assert torch.all(torch.isfinite(X_pred))
        assert not torch.any(torch.isnan(X_pred))
    
    def test_decoder_hidden_dimensions(self):
        """Test decoder with different hidden dimensions."""
        for hidden_dim in [64, 128, 256]:
            decoder = ExpressionDecoder(n_pcs=20, n_genes=200, hidden_dim=hidden_dim)
            X_pca = torch.randn(10, 20)
            
            with torch.no_grad():
                X_pred = decoder(X_pca)
            
            assert X_pred.shape == (10, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
