#!/usr/bin/env python3
"""
Setup script for Music Genre Analysis Project
=============================================

This script automates the setup process for the music genre analysis project.
It checks dependencies, installs required packages, and verifies data availability.

Usage:
    python setup.py [--check-only]

Options:
    --check-only: Only check the current setup without installing anything
"""

import sys
import subprocess
import os
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    print("Checking Python version...")

    if sys.version_info < (3, 8):
        print(f"❌ Error: Python 3.8+ required. Current version: {sys.version}")
        return False

    print(f"✅ Python version: {sys.version.split()[0]} (OK)")
    return True

def check_data_files():
    """Check if required data files exist."""
    print("\nChecking data files...")

    data_dir = Path("Spotify/data")
    required_files = [
        "data.csv",
        "data_w_genres.csv"
    ]

    missing_files = []

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False

    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            # Check file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file}: {size_mb:.1f} MB")

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    return True

def install_requirements():
    """Install required Python packages."""
    print("\nInstalling required packages...")

    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False

    try:
        # Install packages
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)

        print("✅ All packages installed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print(f"Error output: {e.stderr}")
        return False

def verify_installation():
    """Verify that all required packages can be imported."""
    print("\nVerifying package installation...")

    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'plotly',
        'scipy',
    ]

    failed_imports = []

    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            failed_imports.append(package)
            print(f"❌ {package}: {e}")

    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        return False

    print("\n✅ All packages verified!")
    return True

def create_directories():
    """Create necessary output directories."""
    print("\nCreating output directories...")

    directories = [
        "visualizations",
        "clustering_results",
        "results",
        "notebook_results"
    ]

    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created: {dir_name}/")

    return True

def test_basic_functionality():
    """Test basic functionality of the analysis module."""
    print("\nTesting basic functionality...")

    try:
        # Test import
        from music_genre_analysis import MusicGenreAnalyzer
        print("✅ Successfully imported MusicGenreAnalyzer")

        # Test initialization
        analyzer = MusicGenreAnalyzer()
        print("✅ Successfully initialized analyzer")

        return True

    except Exception as e:
        print(f"❌ Error testing functionality: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup script for Music Genre Analysis Project"
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check the current setup without installing anything'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MUSIC GENRE ANALYSIS - SETUP SCRIPT")
    print("=" * 60)

    if args.check_only:
        print("Running setup check only (no installations)...")

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Check data files
    if not check_data_files():
        print("\n⚠️  Data files missing. Please ensure your Spotify dataset is in Spotify/data/")
        if not args.check_only:
            response = input("Continue setup anyway? (y/n): ").lower().strip()
            if response != 'y':
                sys.exit(1)

    # Step 3: Install requirements (if not check-only)
    if not args.check_only:
        if not install_requirements():
            sys.exit(1)

    # Step 4: Verify installation
    if not verify_installation():
        sys.exit(1)

    # Step 5: Create directories (if not check-only)
    if not args.check_only:
        create_directories()

    # Step 6: Test functionality
    if not test_basic_functionality():
        sys.exit(1)

    # Success!
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)

    if not args.check_only:
        print("\nYou can now run the analysis using:")
        print("  • Quick analysis:    python run_analysis.py --quick")
        print("  • Full analysis:     python run_analysis.py --full")
        print("  • With experiments:  python run_analysis.py --experiments")
        print("  • Jupyter notebook:  jupyter notebook music_genre_analysis.ipynb")
    else:
        print("\nSetup check completed. System is ready for analysis!")

    print("\nFor detailed usage instructions, see README.md")

if __name__ == "__main__":
    main()