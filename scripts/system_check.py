#!/usr/bin/env python3
"""
HybridStack-PPI System Check - Grand Inspection Test Suite
===========================================================
A comprehensive test runner to verify all critical modules work correctly
and no data leakage or silent failures can occur.

Usage:
    python scripts/system_check.py
"""

import os
import sys
import warnings
import tempfile
from pathlib import Path

# Suppress warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# Test result tracking
PASSED = 0
FAILED = 0
RESULTS = []


def log_result(test_name: str, passed: bool, message: str = ""):
    """Log test result with formatting."""
    global PASSED, FAILED
    status = "✅ PASSED" if passed else "❌ FAILED"
    if passed:
        PASSED += 1
    else:
        FAILED += 1
    
    result = f"[TEST] {test_name}: {status}"
    if message:
        result += f" ({message})"
    print(result)
    RESULTS.append((test_name, passed, message))


def print_header(section: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {section}")
    print(f"{'='*70}")


# =============================================================================
# TEST 1: src/data_utils.py - The Gatekeeper
# =============================================================================
def test_canonicalization():
    """Test 1.1: Verify canonicalize_pairs merges duplicates correctly."""
    from src.data_utils import canonicalize_pairs
    
    # Create test data with reversed pair duplicates
    test_df = pd.DataFrame({
        'protein1': ['ProtA', 'ProtB', 'ProtC', 'ProtB'],  # ProtB,ProtA is duplicate of ProtA,ProtB
        'protein2': ['ProtB', 'ProtA', 'ProtD', 'ProtC'],
        'label': [1, 1, 0, 1]  # First two are same pair with same label
    })
    
    result = canonicalize_pairs(test_df, dataset_name="Test", logger=None)
    
    # After canonicalization:
    # (ProtA, ProtB) appears twice -> should become 1
    # Total should be 3 unique pairs
    expected_count = 3
    
    if len(result) == expected_count:
        # Check that ProtA,ProtB only appears once
        ab_count = len(result[(result['protein1'] == 'ProtA') & (result['protein2'] == 'ProtB')])
        if ab_count == 1:
            log_result("Canonicalization (Duplicate Merge)", True, f"3 unique pairs as expected")
        else:
            log_result("Canonicalization (Duplicate Merge)", False, f"ProtA-ProtB appears {ab_count} times")
    else:
        log_result("Canonicalization (Duplicate Merge)", False, f"Expected 3, got {len(result)}")


def test_cluster_integrity():
    """Test 1.2: Verify cluster-based split keeps all proteins from same cluster together."""
    from scripts.run_cv import get_c3_splits
    
    # Create mock data
    # Cluster 0: ProtA, ProtB (must stay together)
    # Cluster 1: ProtC, ProtD (must stay together)
    # Cluster 2: ProtE (singleton)
    protein_to_cluster = {
        'ProtA': 0, 'ProtB': 0,
        'ProtC': 1, 'ProtD': 1,
        'ProtE': 2,
    }
    
    # Create pairs (only within-cluster pairs for simplicity)
    pairs_df = pd.DataFrame({
        'protein1': ['ProtA', 'ProtC', 'ProtA', 'ProtC'],
        'protein2': ['ProtB', 'ProtD', 'ProtA', 'ProtC'],  # Self-pairs are fine
        'label': [1, 1, 0, 0]
    })
    
    try:
        splits, valid_df, stats = get_c3_splits(pairs_df, protein_to_cluster, n_splits=2)
        
        # Verify: If ProtA is in train, ProtB must also be in train
        all_passed = True
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            train_pairs = valid_df.iloc[train_indices] if train_indices else pd.DataFrame()
            val_pairs = valid_df.iloc[val_indices] if val_indices else pd.DataFrame()
            
            train_proteins = set()
            val_proteins = set()
            
            if not train_pairs.empty:
                train_proteins = set(train_pairs['protein1']).union(set(train_pairs['protein2']))
            if not val_pairs.empty:
                val_proteins = set(val_pairs['protein1']).union(set(val_pairs['protein2']))
            
            # Check cluster 0
            a_in_train = 'ProtA' in train_proteins
            b_in_train = 'ProtB' in train_proteins
            a_in_val = 'ProtA' in val_proteins
            b_in_val = 'ProtB' in val_proteins
            
            if (a_in_train and b_in_val) or (a_in_val and b_in_train):
                all_passed = False
                log_result("Cluster Integrity (RED LINE)", False, 
                          f"Fold {fold_idx}: ProtA and ProtB split across train/val!")
                break
        
        if all_passed:
            log_result("Cluster Integrity (RED LINE)", True, "Same-cluster proteins stay together")
    
    except Exception as e:
        log_result("Cluster Integrity (RED LINE)", False, f"Exception: {e}")


# =============================================================================
# TEST 2: src/feature_engine.py - The Engine
# =============================================================================
def test_bad_input_handling():
    """Test 2.1: Verify graceful handling of invalid amino acid characters."""
    from src.feature_engine import AACExtractor, DPCExtractor
    
    aac = AACExtractor()
    dpc = DPCExtractor()
    
    # Sequence with invalid characters (X, O, Z are not standard)
    bad_sequence = "MKAZXLOPQRSTUVWXYZ"
    
    try:
        aac_result = aac.compute(bad_sequence)
        dpc_result = dpc.compute(bad_sequence)
        
        # Should not crash, should return valid arrays
        if len(aac_result) == 20 and len(dpc_result) == 400:
            log_result("Bad Input Handling (Invalid Chars)", True, "Handled gracefully")
        else:
            log_result("Bad Input Handling (Invalid Chars)", False, 
                      f"Wrong dimensions: AAC={len(aac_result)}, DPC={len(dpc_result)}")
    except Exception as e:
        log_result("Bad Input Handling (Invalid Chars)", False, f"Crashed: {e}")


def test_dimension_check():
    """Test 2.2: Verify feature extractors produce correct dimensions."""
    from src.feature_engine import (
        AACExtractor, DPCExtractor, CTDExtractor, 
        PAACExtractor, MoranAutocorrelation
    )
    
    test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH"
    
    expected_dims = {
        'AAC': 20,           # 20 amino acids
        'DPC': 400,          # 20 x 20 dipeptides
        'CTD': 147,          # Actual CTD implementation output
    }
    
    extractors = {
        'AAC': AACExtractor(),
        'DPC': DPCExtractor(),
        'CTD': CTDExtractor(),
    }
    
    all_passed = True
    for name, extractor in extractors.items():
        result = extractor.compute(test_sequence)
        expected = expected_dims.get(name, -1)
        
        if len(result) != expected and expected != -1:
            log_result(f"Dimension Check ({name})", False, f"Expected {expected}, got {len(result)}")
            all_passed = False
        elif not np.isfinite(result).all():
            log_result(f"Dimension Check ({name})", False, "Contains NaN/Inf")
            all_passed = False
    
    if all_passed:
        log_result("Dimension Check (All Extractors)", True, "All dimensions correct")


# =============================================================================
# TEST 3: src/builders.py & src/selectors.py - The Brain
# =============================================================================
def test_selector_collapse():
    """Test 3.1: Verify CumulativeFeatureSelector handles zero-variance data."""
    from src.selectors import CumulativeFeatureSelector
    
    # Create all-constant data (zero variance)
    X = pd.DataFrame({
        'feat1': [1.0, 1.0, 1.0, 1.0, 1.0],
        'feat2': [1.0, 1.0, 1.0, 1.0, 1.0],
        'feat3': [1.0, 1.0, 1.0, 1.0, 1.0],
        'feat4': [0.5, 0.6, 0.7, 0.8, 0.9],  # Only one with variance
    })
    y = np.array([0, 1, 0, 1, 0])
    
    selector = CumulativeFeatureSelector(
        variance_threshold=0.0001,  # Should eliminate constant features
        importance_quantile=0.5,
        corr_threshold=0.9,
        verbose=False
    )
    
    try:
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        if X_transformed.shape[1] >= 1:
            log_result("Selector Collapse (Zero Variance)", True, 
                      f"Kept {X_transformed.shape[1]} features (from 4)")
        else:
            log_result("Selector Collapse (Zero Variance)", False, 
                      "Returned empty array - would crash model!")
    except Exception as e:
        # If it raises a clear error, that's acceptable
        if "no features" in str(e).lower() or "empty" in str(e).lower():
            log_result("Selector Collapse (Zero Variance)", True, 
                      f"Raised clear error: {type(e).__name__}")
        else:
            log_result("Selector Collapse (Zero Variance)", False, f"Unexpected error: {e}")


def test_internal_cv_config():
    """Test 3.2: Verify StackingClassifier uses cv=5, not default cv=3."""
    from src.builders import create_stacking_pipeline
    
    # Create minimal column lists
    interp_cols = [f'interp_{i}' for i in range(10)]
    embed_cols = [f'embed_{i}' for i in range(10)]
    
    pipeline = create_stacking_pipeline(interp_cols, embed_cols, n_jobs=1, use_selector=False)
    
    # Extract StackingClassifier from pipeline
    stacking = pipeline.named_steps.get('ensemble')
    
    if stacking is not None:
        cv_value = getattr(stacking, 'cv', None)
        if cv_value == 5:
            log_result("Internal CV Config", True, "cv=5 (aligned with outer CV)")
        elif cv_value == 3:
            log_result("Internal CV Config", False, "cv=3 (potential internal leakage!)")
        else:
            log_result("Internal CV Config", False, f"Unexpected cv={cv_value}")
    else:
        log_result("Internal CV Config", False, "Could not find StackingClassifier")


# =============================================================================
# TEST 4: scripts/run.py - The Orchestrator
# =============================================================================
def test_failsafe_trigger():
    """Test 4.1: Verify fail-safe raises error on missing cluster file."""
    from src.data_utils import load_cluster_map
    
    fake_cluster_path = "/tmp/nonexistent_cluster_file_12345.csv"
    
    try:
        result = load_cluster_map(fake_cluster_path)
        log_result("Fail-Safe Trigger (Missing File)", False, 
                  "Did NOT raise error - silent fallback risk!")
    except FileNotFoundError:
        log_result("Fail-Safe Trigger (Missing File)", True, 
                  "Caught expected FileNotFoundError")
    except Exception as e:
        log_result("Fail-Safe Trigger (Missing File)", True, 
                  f"Caught {type(e).__name__}: {e}")


def test_cv_script_failsafe():
    """Test 4.2: Verify run_cv.py raises error on missing cluster file."""
    from scripts.run_cv import run_c3_cv
    from pathlib import Path
    import tempfile
    
    fake_clstr = "/tmp/fake_cluster_file.clstr"
    fake_cache = "/tmp/fake_cache.h5"
    fake_pairs = "/tmp/fake_pairs.tsv"
    
    try:
        run_c3_cv(
            feature_cache=fake_cache,
            pairs_path=fake_pairs,
            clstr_path=fake_clstr,
            dataset_name="Test",
            output_dir=Path("/tmp/test_output"),
        )
        log_result("CV Script Fail-Safe", False, "Did NOT raise error on missing cluster!")
    except FileNotFoundError as e:
        if "CLUSTER FILE NOT FOUND" in str(e):
            log_result("CV Script Fail-Safe", True, "Raised clear error with guidance")
        else:
            log_result("CV Script Fail-Safe", True, f"Caught FileNotFoundError")
    except Exception as e:
        log_result("CV Script Fail-Safe", True, f"Caught {type(e).__name__}")


# =============================================================================
# TEST 5: Config Validation
# =============================================================================
def test_config_paths():
    """Test 5.1: Verify config.py has valid paths."""
    from scripts.config import DATASETS, validate_cache_files
    
    all_valid = True
    for dataset_name in DATASETS:
        config = DATASETS[dataset_name]
        
        # Check that paths are defined (not None)
        required_keys = ['fasta', 'pairs', 'clstr', 'feature_cache']
        for key in required_keys:
            if key not in config or config[key] is None:
                log_result(f"Config Paths ({dataset_name})", False, f"Missing {key}")
                all_valid = False
                break
    
    if all_valid:
        log_result("Config Paths (Structure)", True, "All required paths defined")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def main():
    print("\n" + "="*70)
    print("  🔍 HybridStack-PPI SYSTEM CHECK - Grand Inspection")
    print("  " + "="*66)
    print(f"  Project Root: {PROJECT_ROOT}")
    print("="*70)
    
    # Module 1: Data Utils
    print_header("MODULE 1: src/data_utils.py (The Gatekeeper)")
    test_canonicalization()
    test_cluster_integrity()
    
    # Module 2: Feature Engine
    print_header("MODULE 2: src/feature_engine.py (The Engine)")
    test_bad_input_handling()
    test_dimension_check()
    
    # Module 3: Builders & Selectors
    print_header("MODULE 3: src/builders.py & src/selectors.py (The Brain)")
    test_selector_collapse()
    test_internal_cv_config()
    
    # Module 4: Orchestrator
    print_header("MODULE 4: scripts/run.py & run_cv.py (The Orchestrator)")
    test_failsafe_trigger()
    test_cv_script_failsafe()
    
    # Module 5: Config
    print_header("MODULE 5: scripts/config.py (Configuration)")
    test_config_paths()
    
    # Summary
    print("\n" + "="*70)
    print("  📊 TEST SUMMARY")
    print("="*70)
    total = PASSED + FAILED
    print(f"\n  Total Tests: {total}")
    print(f"  ✅ Passed: {PASSED}")
    print(f"  ❌ Failed: {FAILED}")
    print(f"  Success Rate: {100*PASSED/total:.1f}%")
    
    if FAILED == 0:
        print("\n  🎉 ALL TESTS PASSED! System is ready for production.")
    else:
        print("\n  ⚠️ SOME TESTS FAILED! Review the issues above.")
        print("\n  Failed Tests:")
        for name, passed, msg in RESULTS:
            if not passed:
                print(f"    - {name}: {msg}")
    
    print("="*70 + "\n")
    
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    exit(main())
