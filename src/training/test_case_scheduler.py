#!/usr/bin/env python3
"""
Test Case Scheduler for RL Training

Provides flexible scheduling of test cases during training with:
- Weighted random sampling based on percentages
- Sequential cycling through test cases
- Reproducible sampling with seeds
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.eval.testcases import TestCase


@dataclass
class TestCaseWeight:
    """Weight configuration for a test case"""
    case_id: str
    weight: float
    description: Optional[str] = None


class TestCaseScheduler:
    """
    Scheduler for selecting test cases during training
    
    Supports two modes:
    1. Weighted random sampling based on percentages
    2. Sequential cycling through test cases
    """
    
    def __init__(self, test_cases: List[TestCase], weights: Optional[List[TestCaseWeight]] = None, 
                 seed: int = 42):
        """
        Initialize scheduler
        
        Args:
            test_cases: List of available test cases
            weights: Optional list of weights for random sampling. If None, uses sequential cycling.
            seed: Random seed for reproducible sampling
        """
        self.test_cases = test_cases
        self.case_id_to_index = {case.id: i for i, case in enumerate(test_cases)}
        self.rng = np.random.RandomState(seed)
        
        if weights is not None:
            self._setup_weighted_sampling(weights)
        else:
            self._setup_sequential_cycling()
    
    def _setup_weighted_sampling(self, weights: List[TestCaseWeight]):
        """Setup weighted random sampling"""
        self.mode = "weighted"
        
        # Validate that all weight case IDs exist
        available_ids = set(case.id for case in self.test_cases)
        weight_ids = set(w.case_id for w in weights)
        
        missing_ids = weight_ids - available_ids
        if missing_ids:
            raise ValueError(f"Weight case IDs not found in test suite: {missing_ids}")
        
        # Create probability distribution
        self.weights = {}
        total_weight = sum(w.weight for w in weights)
        
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        
        for weight in weights:
            self.weights[weight.case_id] = weight.weight / total_weight
        
        # For cases not in weights, assign zero probability
        for case in self.test_cases:
            if case.id not in self.weights:
                self.weights[case.id] = 0.0
        
        # Create arrays for numpy random choice
        self.case_ids = [case.id for case in self.test_cases]
        self.probabilities = [self.weights[case_id] for case_id in self.case_ids]
        
        print(f"Weighted sampling initialized:")
        for case_id, prob in zip(self.case_ids, self.probabilities):
            if prob > 0:
                print(f"  {case_id}: {prob:.1%}")
    
    def _setup_sequential_cycling(self):
        """Setup sequential cycling through test cases"""
        self.mode = "sequential"
        self.current_index = 0
        
        print(f"Sequential cycling initialized with {len(self.test_cases)} test cases:")
        for i, case in enumerate(self.test_cases):
            print(f"  {i}: {case.id}")
    
    def get_next_test_case(self, episode: int) -> TestCase:
        """
        Get the next test case for the given episode
        
        Args:
            episode: Current episode number
            
        Returns:
            Selected test case
        """
        if self.mode == "weighted":
            return self._get_weighted_test_case(episode)
        else:
            return self._get_sequential_test_case(episode)
    
    def _get_weighted_test_case(self, episode: int) -> TestCase:
        """Get test case using weighted random sampling"""
        # Use episode number to advance RNG state for reproducibility
        # This ensures the same episode always gets the same test case
        temp_rng = np.random.RandomState(self.rng.get_state()[1][0] + episode)
        
        chosen_index = temp_rng.choice(len(self.test_cases), p=self.probabilities)
        return self.test_cases[chosen_index]
    
    def _get_sequential_test_case(self, episode: int) -> TestCase:
        """Get test case using sequential cycling"""
        index = episode % len(self.test_cases)
        return self.test_cases[index]
    
    def get_statistics(self, num_episodes: int) -> Dict[str, Any]:
        """
        Get statistics about test case distribution over a number of episodes
        
        Args:
            num_episodes: Number of episodes to simulate
            
        Returns:
            Dictionary with statistics
        """
        case_counts = {case.id: 0 for case in self.test_cases}
        
        for episode in range(num_episodes):
            test_case = self.get_next_test_case(episode)
            case_counts[test_case.id] += 1
        
        case_percentages = {case_id: count / num_episodes * 100 
                           for case_id, count in case_counts.items()}
        
        return {
            'mode': self.mode,
            'num_episodes': num_episodes,
            'case_counts': case_counts,
            'case_percentages': case_percentages,
            'total_cases': len(self.test_cases),
            'active_cases': sum(1 for count in case_counts.values() if count > 0)
        }


def load_scheduler_config(config_path: str) -> List[TestCaseWeight]:
    """
    Load scheduler configuration from JSON file
    
    Expected format:
    {
        "description": "Training schedule for basic scenarios",
        "weights": [
            {"case_id": "center_to_corner_NE_facing_north", "weight": 0.4, "description": "Primary training case"},
            {"case_id": "food_behind_north", "weight": 0.3, "description": "Challenging case"},
            {"case_id": "simple_forward", "weight": 0.3, "description": "Easy case"}
        ]
    }
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        List of TestCaseWeight objects
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'weights' not in config:
        raise ValueError("Configuration file must contain 'weights' field")
    
    weights = []
    for weight_config in config['weights']:
        if 'case_id' not in weight_config or 'weight' not in weight_config:
            raise ValueError("Each weight must have 'case_id' and 'weight' fields")
        
        weights.append(TestCaseWeight(
            case_id=weight_config['case_id'],
            weight=float(weight_config['weight']),
            description=weight_config.get('description')
        ))
    
    return weights


def create_example_scheduler_config(output_path: str, test_cases: List[TestCase]):
    """
    Create an example scheduler configuration file
    
    Args:
        output_path: Path where to save the example config
        test_cases: List of test cases to include in the example
    """
    # Create equal weights for all test cases
    equal_weight = 1.0 / len(test_cases)
    
    config = {
        "description": f"Example scheduler config with {len(test_cases)} test cases",
        "weights": []
    }
    
    for case in test_cases:
        config["weights"].append({
            "case_id": case.id,
            "weight": equal_weight,
            "description": f"Test case: {case.id}"
        })
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Example scheduler config saved to: {output_path}")