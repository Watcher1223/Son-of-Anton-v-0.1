"""
Unit tests for EndpointFeatureExtractor
"""

import pytest
from src.extractors.endpoint_features import EndpointFeatureExtractor


def test_extractor_initialization():
    """Test EndpointFeatureExtractor initializes correctly"""
    extractor = EndpointFeatureExtractor()
    assert extractor is not None


def test_extract_path_features():
    """Test path feature extraction"""
    extractor = EndpointFeatureExtractor()
    
    # Test simple path
    features = extractor.extract_path_features('/api/users')
    assert features['path_depth'] == 2
    assert features['has_api_prefix'] == True
    assert features['path_param_count'] == 0
    
    # Test path with parameters
    features = extractor.extract_path_features('/api/users/:id')
    assert features['path_depth'] == 3
    assert features['path_param_count'] == 1
    assert features['has_path_params'] == True


def test_extract_method_features():
    """Test HTTP method feature extraction"""
    extractor = EndpointFeatureExtractor()
    
    # Test GET
    features = extractor.extract_method_features('GET')
    assert features['method'] == 'GET'
    assert features['is_get'] == True
    assert features['is_safe_method'] == True
    assert features['modifies_data'] == False
    
    # Test POST
    features = extractor.extract_method_features('POST')
    assert features['method'] == 'POST'
    assert features['is_post'] == True
    assert features['is_safe_method'] == False
    assert features['modifies_data'] == True


def test_extract_method_features_case_insensitive():
    """Test method extraction handles lowercase"""
    extractor = EndpointFeatureExtractor()
    
    features = extractor.extract_method_features('post')
    assert features['method'] == 'POST'
    assert features['is_post'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

