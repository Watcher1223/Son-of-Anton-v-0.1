"""
Unit tests for SchemaDetector
"""

import pytest
from pathlib import Path
from src.detectors.schema_detector import SchemaDetector


def test_schema_detector_initialization():
    """Test SchemaDetector initializes correctly"""
    detector = SchemaDetector()
    assert detector is not None
    assert hasattr(detector, 'openapi_patterns')
    assert hasattr(detector, 'validator_patterns')
    assert hasattr(detector, 'typedef_patterns')


def test_detect_openapi_files_empty():
    """Test detection on non-existent path"""
    detector = SchemaDetector()
    fake_path = Path("/nonexistent/path")
    files = detector.detect_openapi_files(fake_path)
    assert isinstance(files, list)
    assert len(files) == 0


def test_detect_validator_libraries_no_package_json():
    """Test validator library detection with missing package.json"""
    detector = SchemaDetector()
    fake_path = Path("/nonexistent/path")
    libs = detector.detect_validator_libraries(fake_path)
    assert isinstance(libs, list)
    assert len(libs) == 0


def test_classify_schema_type_unknown():
    """Test schema classification on non-existent path"""
    detector = SchemaDetector()
    fake_path = Path("/nonexistent/path")
    schema_type, details = detector.classify_schema_type(fake_path)
    assert schema_type == 'unknown'
    assert isinstance(details, dict)
    assert details['openapi_count'] == 0
    assert details['validator_count'] == 0
    assert details['typedef_count'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

