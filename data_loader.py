"""
data_loader.py - download and load the raw IMDb 50K Movie Reviews dataset.

The raw data is NOT stored in this repository. It is downloaded from Kaggle
with kagglehub the first time, and kagglehub keeps a cached copy afterwards.

Owner: Member 3

Run on its own:
    python data_loader.py
"""

import os

import kagglehub
import pandas as pd

DATASET_HANDLE = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"


def load_imdb_data():
    """Download (if needed) and load the IMDb dataset as a pandas DataFrame."""
    path = kagglehub.dataset_download(DATASET_HANDLE)
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    csv_path = os.path.join(path, csv_files[0])
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    df = load_imdb_data()
    print("Loaded IMDb dataset")
    print("Shape (rows, columns):", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nTarget distribution (sentiment):")
    print(df["sentiment"].value_counts())
