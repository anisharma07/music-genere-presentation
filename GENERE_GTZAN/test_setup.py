"""
Test Script for Music Genre Discovery Project
==============================================

This script tests all modules to ensure they work correctly.

Author: Anirudh Sharma
Date: November 2025
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """Test if all required packages are installed."""
    print("\n" + "=" * 80)
    print("TEST 1: Checking Package Imports")
    print("=" * 80)

    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'sklearn': 'sklearn',
        'scipy': 'scipy'
    }

    failed = []

    for name, import_path in required_packages.items():
        try:
            __import__(import_path)
            print(f"  ✓ {name:<15} - OK")
        except ImportError:
            print(f"  ✗ {name:<15} - MISSING")
            failed.append(name)

    if failed:
        print(f"\n  ⚠ Missing packages: {', '.join(failed)}")
        print("  → Install with: pip install " + " ".join(failed))
        return False
    else:
        print("\n  ✓ All packages installed!")
        return True


def test_dataset():
    """Test if dataset files exist."""
    print("\n" + "=" * 80)
    print("TEST 2: Checking Dataset Files")
    print("=" * 80)

    required_files = [
        'gtzan/features_30_sec.csv',
        'gtzan/features_3_sec.csv',
        'gtzan/gtzan_metadata.csv'
    ]

    missing = []

    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath:<35} - EXISTS")
        else:
            print(f"  ✗ {filepath:<35} - MISSING")
            missing.append(filepath)

    if missing:
        print(f"\n  ⚠ Missing files: {len(missing)}")
        print("  → Ensure GTZAN dataset is in the gtzan/ directory")
        return False
    else:
        print("\n  ✓ All dataset files present!")
        return True


def test_data_loading():
    """Test if data can be loaded."""
    print("\n" + "=" * 80)
    print("TEST 3: Loading Dataset")
    print("=" * 80)

    try:
        import pandas as pd

        df = pd.read_csv('gtzan/features_30_sec.csv')

        print(f"  ✓ Dataset loaded successfully")
        print(f"  - Shape: {df.shape}")
        print(f"  - Samples: {len(df)}")
        print(f"  - Features: {df.shape[1]}")
        print(f"  - Genres: {df['label'].nunique()}")

        # Check for basic columns
        if 'label' in df.columns and 'filename' in df.columns:
            print(f"  ✓ Required columns present")
        else:
            print(f"  ✗ Missing required columns")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error loading dataset: {str(e)}")
        return False


def test_modules():
    """Test if project modules can be imported."""
    print("\n" + "=" * 80)
    print("TEST 4: Checking Project Modules")
    print("=" * 80)

    modules = [
        'data_analysis',
        'clustering_implementation',
        'cross_validation',
        'utils',
        'config'
    ]

    failed = []

    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module:<30} - OK")
        except Exception as e:
            print(f"  ✗ {module:<30} - ERROR: {str(e)[:40]}")
            failed.append(module)

    if failed:
        print(f"\n  ⚠ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n  ✓ All modules imported successfully!")
        return True


def test_directories():
    """Test if necessary directories exist or create them."""
    print("\n" + "=" * 80)
    print("TEST 5: Checking/Creating Directories")
    print("=" * 80)

    directories = ['results', 'visualizations', 'reports']

    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✓ {directory}/ - EXISTS")
        else:
            try:
                os.makedirs(directory)
                print(f"  ✓ {directory}/ - CREATED")
            except Exception as e:
                print(f"  ✗ {directory}/ - ERROR: {str(e)}")
                return False

    print("\n  ✓ All directories ready!")
    return True


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n" + "=" * 80)
    print("TEST 6: Testing Basic Functionality")
    print("=" * 80)

    try:
        # Test data analysis
        print("\n  Testing data_analysis module...")
        from data_analysis import MusicDataAnalyzer

        analyzer = MusicDataAnalyzer('gtzan/features_30_sec.csv')
        print("    ✓ MusicDataAnalyzer initialized")

        # Test clustering
        print("\n  Testing clustering_implementation module...")
        from clustering_implementation import MusicGenreClusterer

        clusterer = MusicGenreClusterer('gtzan/features_30_sec.csv')
        print("    ✓ MusicGenreClusterer initialized")
        print(
            f"    - PCA variance explained: {clusterer.pca.explained_variance_ratio_.sum():.2%}")

        # Test config
        print("\n  Testing config module...")
        from config import get_config_summary

        config = get_config_summary()
        print("    ✓ Configuration loaded")

        print("\n  ✓ Basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"\n  ✗ Error in functionality test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_clustering():
    """Test a quick clustering run with minimal data."""
    print("\n" + "=" * 80)
    print("TEST 7: Quick Clustering Test")
    print("=" * 80)

    try:
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Load data
        df = pd.read_csv('gtzan/features_30_sec.csv')

        # Sample subset for quick test
        df_sample = df.sample(n=min(100, len(df)), random_state=42)

        # Prepare features
        feature_cols = [col for col in df_sample.columns
                        if col not in ['filename', 'label']]
        X = df_sample[feature_cols].values

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=10, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Cluster
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=5)
        labels = kmeans.fit_predict(X_pca)

        print(f"  ✓ Quick clustering test completed")
        print(f"  - Samples: {len(df_sample)}")
        print(f"  - Features: {X.shape[1]} → {X_pca.shape[1]} (PCA)")
        print(f"  - Clusters found: {len(np.unique(labels))}")
        print(f"  - PCA variance: {pca.explained_variance_ratio_.sum():.2%}")

        print("\n  ✓ Clustering pipeline works correctly!")
        return True

    except Exception as e:
        print(f"\n  ✗ Error in clustering test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" " * 20 + "MUSIC GENRE DISCOVERY PROJECT")
    print(" " * 25 + "TEST SUITE")
    print("=" * 80)

    tests = [
        ("Package Imports", test_imports),
        ("Dataset Files", test_dataset),
        ("Data Loading", test_data_loading),
        ("Project Modules", test_modules),
        ("Directories", test_directories),
        ("Basic Functionality", test_basic_functionality),
        ("Quick Clustering", test_quick_clustering)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n  ✗ Test '{test_name}' crashed: {str(e)}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {test_name}")

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 All tests passed! System is ready to run.")
        print("\nYou can now run:")
        print("  python main.py")
        return True
    else:
        print(
            f"\n⚠ {total - passed} test(s) failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    """
    Run the test suite.
    """
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test suite crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
