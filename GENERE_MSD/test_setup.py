"""
Test script to verify the setup and run a small sample
"""

import os
import sys


def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - MISSING")
        return False

    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - MISSING")
        return False

    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError:
        print("  ✗ scikit-learn - MISSING")
        return False

    try:
        import h5py
        print("  ✓ h5py")
    except ImportError:
        print("  ✗ h5py - MISSING")
        return False

    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib - MISSING")
        return False

    try:
        import seaborn
        print("  ✓ seaborn")
    except ImportError:
        print("  ✗ seaborn - MISSING")
        return False

    try:
        import scipy
        print("  ✓ scipy")
    except ImportError:
        print("  ✗ scipy - MISSING")
        return False

    try:
        from tqdm import tqdm
        print("  ✓ tqdm")
    except ImportError:
        print("  ✗ tqdm - MISSING")
        return False

    try:
        import joblib
        print("  ✓ joblib")
    except ImportError:
        print("  ✗ joblib - MISSING")
        return False

    print("\n✓ All packages installed!\n")
    return True


def test_dataset():
    """Test if dataset is accessible"""
    print("Testing dataset access...")

    from config import DATA_DIR

    if not os.path.exists(DATA_DIR):
        print(f"  ✗ Dataset directory not found: {DATA_DIR}")
        return False

    print(f"  ✓ Dataset directory exists: {DATA_DIR}")

    # Try to find some HDF5 files
    h5_count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.h5'):
                h5_count += 1
                if h5_count >= 5:
                    break
        if h5_count >= 5:
            break

    if h5_count == 0:
        print(f"  ✗ No HDF5 files found in {DATA_DIR}")
        return False

    print(f"  ✓ Found HDF5 files (sample: {h5_count})")
    print()
    return True


def test_modules():
    """Test if custom modules can be imported"""
    print("Testing custom modules...")

    try:
        import config
        print("  ✓ config.py")
    except ImportError as e:
        print(f"  ✗ config.py - {e}")
        return False

    try:
        from feature_extractor import MillionSongFeatureExtractor
        print("  ✓ feature_extractor.py")
    except ImportError as e:
        print(f"  ✗ feature_extractor.py - {e}")
        return False

    try:
        from data_cleaner import DataCleaner
        print("  ✓ data_cleaner.py")
    except ImportError as e:
        print(f"  ✗ data_cleaner.py - {e}")
        return False

    try:
        from clustering import MusicGenreClusterer
        print("  ✓ clustering.py")
    except ImportError as e:
        print(f"  ✗ clustering.py - {e}")
        return False

    try:
        from evaluation import ClusteringEvaluator
        print("  ✓ evaluation.py")
    except ImportError as e:
        print(f"  ✗ evaluation.py - {e}")
        return False

    try:
        from visualization import ClusteringVisualizer
        print("  ✓ visualization.py")
    except ImportError as e:
        print(f"  ✗ visualization.py - {e}")
        return False

    print("\n✓ All modules can be imported!\n")
    return True


def run_mini_test():
    """Run a mini test with a few files"""
    print("="*60)
    print("Running mini test (10 files)...")
    print("="*60)
    print()

    from feature_extractor import MillionSongFeatureExtractor
    from config import DATA_DIR, OUTPUT_DIR

    # Extract features from 10 files
    extractor = MillionSongFeatureExtractor(DATA_DIR)
    df = extractor.extract_all_features(max_files=10)

    if len(df) == 0:
        print("✗ No features extracted")
        return False

    print(f"\n✓ Successfully extracted features from {len(df)} files")
    print(f"✓ Features shape: {df.shape}")
    print(f"\nSample features:")
    print(df.head())

    # Save test output
    test_output = os.path.join(OUTPUT_DIR, 'test_features.csv')
    df.to_csv(test_output, index=False)
    print(f"\n✓ Test output saved to: {test_output}")

    return True


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("Music Genre Discovery - Setup Verification")
    print("="*60)
    print()

    # Test 1: Imports
    if not test_imports():
        print("\n✗ Import test failed. Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Test 2: Dataset
    dataset_ok = test_dataset()
    if not dataset_ok:
        print("⚠ Dataset test failed. Pipeline will not run without dataset.")
        print("  However, you can still test the code structure.\n")

    # Test 3: Modules
    if not test_modules():
        print("\n✗ Module test failed. Check for syntax errors in code files.")
        sys.exit(1)

    # Test 4: Mini run (only if dataset is accessible)
    if dataset_ok:
        response = input("\nRun mini test with 10 files? (y/n): ")
        if response.lower() == 'y':
            if run_mini_test():
                print("\n" + "="*60)
                print("✓ ALL TESTS PASSED!")
                print("="*60)
                print("\nYou're ready to run the full pipeline:")
                print("  python main.py")
                print("\nOr use the quick start script:")
                print("  ./run.sh")
                print()
            else:
                print("\n✗ Mini test failed")
                sys.exit(1)
        else:
            print("\n✓ Setup verified (skipped mini test)")
    else:
        print("\n✓ Code setup verified")
        print("⚠ Add dataset to run the pipeline")


if __name__ == "__main__":
    main()
