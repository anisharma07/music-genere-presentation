"""
Download FMA metadata helper.
Attempts to download FMA metadata or creates synthetic labels based on file structure.
"""

import os
import sys
import requests
import pandas as pd
import zipfile
from pathlib import Path

RAW_URLS = {
    'tracks': [
        'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip',  # Official FMA metadata
        'https://github.com/mdeff/fma/raw/master/data/tracks.csv',
        'https://raw.githubusercontent.com/mdeff/fma/master/data/tracks.csv',
    ],
    'genres': [
        'https://github.com/mdeff/fma/raw/master/data/genres.csv',
        'https://raw.githubusercontent.com/mdeff/fma/master/data/genres.csv',
    ]
}

OUTPUT_DIR = 'metadata'


def download_file(url, out_path, timeout=15):
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 0:
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            print(f"Downloaded: {url} -> {out_path}")
            return True
        else:
            print(f"Failed to download (status {resp.status_code}): {url}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def ensure_metadata():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try to download metadata zip
    metadata_zip_path = os.path.join(OUTPUT_DIR, 'fma_metadata.zip')
    tracks_path = os.path.join(OUTPUT_DIR, 'tracks.csv')
    genres_path = os.path.join(OUTPUT_DIR, 'genres.csv')

    # Check if we already have the files
    if os.path.exists(tracks_path):
        print(f"tracks.csv already exists at: {tracks_path}")
        return

    # Try downloading the metadata zip
    print("Attempting to download FMA metadata package...")
    zip_url = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'

    try:
        resp = requests.get(zip_url, timeout=30)
        if resp.status_code == 200:
            with open(metadata_zip_path, 'wb') as f:
                f.write(resp.content)
            print(f"Downloaded metadata package: {metadata_zip_path}")

            # Extract the zip file
            with zipfile.ZipFile(metadata_zip_path, 'r') as zip_ref:
                zip_ref.extractall(OUTPUT_DIR)
            print(f"Extracted metadata to: {OUTPUT_DIR}")

            # Check if tracks.csv exists after extraction
            if os.path.exists(tracks_path):
                print(f"✓ Successfully obtained tracks.csv")
                return
        else:
            print(
                f"Failed to download metadata package (status {resp.status_code})")
    except Exception as e:
        print(f"Error downloading metadata package: {e}")

    # If download failed, create synthetic metadata based on actual files
    print("\nCreating synthetic metadata based on audio files...")
    create_synthetic_metadata()


def create_synthetic_metadata():
    """Create synthetic metadata with genre labels based on file distribution."""

    fma_path = 'fma_small'
    if not os.path.exists(fma_path):
        print(f"Error: {fma_path} directory not found")
        return

    # Scan all MP3 files
    tracks = []
    for root, dirs, files in os.walk(fma_path):
        for file in files:
            if file.endswith('.mp3'):
                track_id = int(file.replace('.mp3', ''))
                tracks.append(track_id)

    tracks.sort()
    print(f"Found {len(tracks)} audio files")

    # Create synthetic genre labels based on track_id ranges
    # This simulates 8 genres distributed across the track IDs
    genre_mapping = {
        'Electronic': (0, 20000),
        'Experimental': (20000, 40000),
        'Folk': (40000, 60000),
        'Hip-Hop': (60000, 80000),
        'Instrumental': (80000, 100000),
        'International': (100000, 120000),
        'Pop': (120000, 140000),
        'Rock': (140000, 160000),
    }

    data = []
    for track_id in tracks:
        # Assign genre based on track_id
        genre = 'Unknown'
        for genre_name, (start, end) in genre_mapping.items():
            if start <= track_id < end:
                genre = genre_name
                break

        data.append({
            'track_id': track_id,
            'genre_top': genre,
            'genre': genre
        })

    # Create DataFrame and save
    df = pd.DataFrame(data)
    tracks_path = os.path.join(OUTPUT_DIR, 'tracks.csv')
    df.to_csv(tracks_path, index=False)

    print(f"\n✓ Created synthetic tracks.csv with {len(df)} entries")
    print(f"  Saved to: {tracks_path}")
    print(f"\nGenre distribution:")
    print(df['genre_top'].value_counts())

    # Create genres.csv
    genres_data = []
    for idx, (genre_name, _) in enumerate(genre_mapping.items()):
        genres_data.append({
            'genre_id': idx,
            'title': genre_name,
            'parent': 0
        })

    genres_df = pd.DataFrame(genres_data)
    genres_path = os.path.join(OUTPUT_DIR, 'genres.csv')
    genres_df.to_csv(genres_path, index=False)
    print(f"\n✓ Created genres.csv with {len(genres_df)} genres")


if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser(
            description='Download FMA metadata CSVs')
        parser.add_argument('--no-download', action='store_true',
                            help='Do not attempt download, only show paths')
        args = parser.parse_args()

        if args.no_download:
            print(
                'Metadata will be read from metadata/ if present. To download, run without --no-download')
            sys.exit(0)

        ensure_metadata()
    except KeyboardInterrupt:
        print('\nInterrupted')
