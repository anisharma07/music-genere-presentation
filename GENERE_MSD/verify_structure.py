#!/usr/bin/env python3
"""
Project Structure Verification
Checks that all required files are present
"""

import os
import sys


def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    return exists


def main():
    print("\n" + "="*60)
    print("Project Structure Verification")
    print("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_files_exist = True

    print("\n📁 Core Python Modules:")
    all_files_exist &= check_file_exists(
        os.path.join(base_dir, "config.py"), "config.py")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "feature_extractor.py"), "feature_extractor.py")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "data_cleaner.py"), "data_cleaner.py")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "clustering.py"), "clustering.py")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "evaluation.py"), "evaluation.py")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "visualization.py"), "visualization.py")
    all_files_exist &= check_file_exists(
        os.path.join(base_dir, "main.py"), "main.py")

    print("\n📁 Utility Scripts:")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "test_setup.py"), "test_setup.py")
    all_files_exist &= check_file_exists(
        os.path.join(base_dir, "run.sh"), "run.sh")

    print("\n📁 Documentation:")
    all_files_exist &= check_file_exists(
        os.path.join(base_dir, "README.md"), "README.md")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "QUICK_START.md"), "QUICK_START.md")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "FILES_SUMMARY.md"), "FILES_SUMMARY.md")

    print("\n📁 Configuration:")
    all_files_exist &= check_file_exists(os.path.join(
        base_dir, "requirements.txt"), "requirements.txt")

    print("\n📁 Dataset:")
    dataset_dir = os.path.join(
        base_dir, "million song", "millionsongsubset", "MillionSongSubset")
    dataset_exists = os.path.exists(dataset_dir)
    status = "✓" if dataset_exists else "⚠"
    print(f"  {status} Million Song Dataset ({dataset_dir})")
    if not dataset_exists:
        print(f"     Note: Dataset not found. Add it to run the pipeline.")

    print("\n" + "="*60)
    if all_files_exist:
        print("✓ All code files present!")
        if dataset_exists:
            print("✓ Dataset found!")
            print("\nYou're ready to run the pipeline:")
            print("  ./run.sh")
            print("  OR")
            print("  python main.py")
        else:
            print("⚠ Dataset not found (optional for code verification)")
            print("\nYou can test the code structure:")
            print("  python test_setup.py")
    else:
        print("✗ Some files are missing!")
        print("Please ensure all files are present.")
        sys.exit(1)

    print("="*60)
    print()


if __name__ == "__main__":
    main()
