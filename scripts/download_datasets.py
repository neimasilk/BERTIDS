#!/usr/bin/env python3
"""
Download CICIDS2017 dataset for BERT-IDS research.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse

# Dataset URLs (these are example URLs - you'll need to get actual URLs)
DATASETS = {
    'cicids2017': {
        'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
        'description': 'CICIDS2017 Dataset',
        'files': [
            # Add actual download URLs here
            # 'https://example.com/cicids2017_monday.csv',
            # 'https://example.com/cicids2017_tuesday.csv',
        ]
    }
}

def download_file(url, destination, chunk_size=8192):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {destination}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted: {zip_path}")
        return True
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False

def download_cicids2017(data_dir):
    """Download CICIDS2017 dataset."""
    print("🔄 Downloading CICIDS2017 dataset...")
    print("📋 Note: You need to manually download CICIDS2017 from:")
    print("   https://www.unb.ca/cic/datasets/ids-2017.html")
    print("   Please place the CSV files in: data/raw/cicids2017/")
    
    # Create directory structure
    cicids_dir = data_dir / "raw" / "cicids2017"
    cicids_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a README with download instructions
    readme_content = """# CICIDS2017 Dataset

## Download Instructions:

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the following files:
   - Monday-WorkingHours.pcap_ISCX.csv
   - Tuesday-WorkingHours.pcap_ISCX.csv
   - Wednesday-workingHours.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   - Friday-WorkingHours-Morning.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

3. Place all CSV files in this directory: data/raw/cicids2017/

## Dataset Information:
- Total size: ~7GB
- Format: CSV files
- Features: 78 network flow features
- Classes: Normal traffic + 14 attack types
"""
    
    with open(cicids_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"📁 Created directory: {cicids_dir}")
    print(f"📄 Created README with instructions: {cicids_dir}/README.md")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download datasets for BERT-IDS")
    parser.add_argument(
        "--dataset", 
        choices=["cicids2017", "all"], 
        default="cicids2017",
        help="Dataset to download"
    )
    parser.add_argument(
        "--data-dir", 
        type=Path, 
        default=Path("data"),
        help="Data directory path"
    )
    
    args = parser.parse_args()
    
    # Create data directory structure
    args.data_dir.mkdir(exist_ok=True)
    (args.data_dir / "raw").mkdir(exist_ok=True)
    (args.data_dir / "processed").mkdir(exist_ok=True)
    (args.data_dir / "external").mkdir(exist_ok=True)
    
    print(f"🚀 Starting dataset download to: {args.data_dir}")
    
    if args.dataset == "cicids2017" or args.dataset == "all":
        download_cicids2017(args.data_dir)
    
    print("✅ Dataset download setup completed!")
    print("📋 Next steps:")
    print("   1. Manually download the CICIDS2017 files as instructed")
    print("   2. Run: python scripts/preprocess_data.py")
    print("   3. Start data exploration with Jupyter notebooks")

if __name__ == "__main__":
    main()