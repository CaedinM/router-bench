#!/usr/bin/env python3
"""Compatibility entry point for the benchmark uploader."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routing.upload_dataset import upload_dataset


if __name__ == '__main__':
    print(upload_dataset(reuse_existing=False), flush=True)
