#!/usr/bin/env python3
"""
Test suite loader and validation
"""

import json
import os
from typing import List, Optional
from .testcases import TestSuite, TestCase


def load_suite(path: str) -> TestSuite:
    """
    Load test suite from JSON file
    
    Args:
        path: Path to JSON file containing test suite
        
    Returns:
        TestSuite object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or validation fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test suite file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in test suite file: {e}")
    
    try:
        suite = TestSuite.from_dict(data)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid test suite format: {e}")
    
    # Validate the loaded suite
    errors = suite.validate()
    if errors:
        error_msg = "Test suite validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_msg)
    
    return suite


def save_suite(suite: TestSuite, path: str) -> None:
    """
    Save test suite to JSON file
    
    Args:
        suite: TestSuite to save
        path: Output file path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(suite.to_dict(), f, indent=2)


def list_available_suites(suite_dir: str = "data/test_suites") -> List[str]:
    """
    List available test suite files
    
    Args:
        suite_dir: Directory containing test suite files
        
    Returns:
        List of available suite file paths
    """
    if not os.path.exists(suite_dir):
        return []
    
    suites = []
    for filename in os.listdir(suite_dir):
        if filename.endswith('.json'):
            suites.append(os.path.join(suite_dir, filename))
    
    return sorted(suites)


def get_suite_info(path: str) -> Optional[dict]:
    """
    Get basic info about a test suite without fully loading it
    
    Args:
        path: Path to test suite file
        
    Returns:
        Dict with suite_id, version, description, num_cases, or None if invalid
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        return {
            'suite_id': data.get('suite_id', 'unknown'),
            'version': data.get('version', 'unknown'),
            'description': data.get('description', ''),
            'num_cases': len(data.get('test_cases', []))
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None