"""
CNNDetection Dataset Download and Corruption Script
Downloads CNNDetection dataset, extracts it, and applies corruptions to corrupted_data folder
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

# Add data_download to path to import modules
sys.path.insert(0, str(Path(__file__).parent / 'data_download'))

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "tqdm"])
    import requests
    from tqdm import tqdm

from apply_corruptions import ImageCorruptor
from process_dataset import DatasetProcessor


def download_file(url, destination):
    """Download a file from URL with progress bar"""
    print(f"\nDownloading {destination.name}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def download_and_extract_cnndetection(output_dir):
    """Download and extract CNNDetection dataset from Hugging Face"""
    print("=" * 80)
    print("Downloading CNNDetection/ForenSynths dataset from Hugging Face...")
    print("=" * 80)

    dataset_dir = output_dir / "dataset" / "test"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/"
    files = [
        "CNN_synth_testset.zip",
        "progan_testset.zip"
    ]

    try:
        for filename in files:
            url = base_url + filename
            destination = dataset_dir / filename

            download_file(url, destination)

            print(f"Extracting {filename}...")
            with zipfile.ZipFile(destination, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)

            destination.unlink()
            print(f"✓ {filename} extracted and removed")

        print("\n✓ CNNDetection download and extraction complete!")
        return dataset_dir
    except Exception as e:
        print(f"Error: {e}")
        print("Please try manual download from: https://huggingface.co/datasets/sywang/CNNDetection")
        return None


def apply_corruptions(dataset_dir, output_dir, gan_type=None, severity=3, num_workers=None):
    """Apply corruptions to downloaded dataset"""
    print("\n" + "=" * 80)
    if gan_type:
        print(f"Applying corruptions to {gan_type} dataset (severity={severity})...")
    else:
        print(f"Applying corruptions to all GAN types (severity={severity})...")
    print("=" * 80)

    # Create processor with corrupted_data output directory
    processor = DatasetProcessor(output_dir=str(output_dir))

    # Process CNNDetection dataset
    processor.process_cnndetection_dataset(
        dataset_root=dataset_dir.parent.parent,  # Go up to "dataset" root
        gan_type=gan_type,
        severity=severity,
        num_workers=num_workers
    )

    print(f"\n✓ Corruptions applied successfully to {output_dir}!")
    return True


def main():
    print("=" * 80)
    print("CNNDetection Dataset - Download, Extract, and Apply Corruptions")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Download CNNDetection dataset from Hugging Face")
    print("  2. Automatically extract all compressed files")
    print("  3. Apply 6 corruption types (contrast, gaussian_noise, motion_blur,")
    print("     pixelate, jpeg_compression, fog)")
    print("  4. Save corrupted images to 'corrupted_data' folder")
    print("\nAvailable GAN types in CNNDetection:")
    print("  - progan (Progressive GAN)")
    print("  - stylegan (StyleGAN)")
    print("  - stylegan2 (StyleGAN2)")
    print("  - biggan (BigGAN)")
    print("  - cyclegan (CycleGAN)")
    print("  - stargan (StarGAN)")
    print("  - gaugan (GauGAN)")
    print("  - crn (CRN)")
    print("  - imle (IMLE)")
    print("  - sn_gan (SN-GAN)")
    print("  - deepfake (Deepfake)")
    print("  - whichfaceisreal (Which Face is Real)")
    print()

    choice = input("Process all GAN types or specific type? (all/specific): ").strip().lower()

    # Set base output directory to corrupted_data
    base_dir = Path(__file__).parent
    output_dir = base_dir / "corrupted_data"

    # Automatically set corruption severity to 3
    severity = 3
    print(f"\nCorruption severity: {severity} (automatic)")

    num_workers_input = input("Enter number of parallel workers (default=auto): ").strip()
    num_workers = int(num_workers_input) if num_workers_input else None

    # Download and extract
    print("\n" + "=" * 80)
    print("Step 1: Downloading and Extracting Dataset")
    print("=" * 80)
    dataset_dir = download_and_extract_cnndetection(base_dir)

    if not dataset_dir:
        print("\nDownload failed. Exiting.")
        sys.exit(1)

    # Determine GAN type
    gan_type = None
    if choice == 'all':
        gan_type = None
    elif choice == 'specific':
        print("\nAvailable GAN types:")
        gan_types = [
            "progan", "stylegan", "stylegan2", "biggan",
            "cyclegan", "stargan", "gaugan", "crn",
            "imle", "sn_gan", "deepfake", "whichfaceisreal"
        ]
        for i, gan in enumerate(gan_types, 1):
            print(f"  {i}. {gan}")

        gan_choice = input("\nEnter GAN type name or number: ").strip()

        # Handle numeric input
        if gan_choice.isdigit():
            idx = int(gan_choice) - 1
            if 0 <= idx < len(gan_types):
                gan_type = gan_types[idx]
            else:
                print("Invalid number. Exiting.")
                sys.exit(1)
        else:
            gan_type = gan_choice.lower()
            if gan_type not in gan_types:
                print(f"Invalid GAN type: {gan_type}. Exiting.")
                sys.exit(1)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Apply corruptions
    print("\n" + "=" * 80)
    print("Step 2: Applying Corruptions")
    print("=" * 80)

    success = apply_corruptions(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        gan_type=gan_type,
        severity=severity,
        num_workers=num_workers
    )

    if not success:
        print("\nCorruption failed. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("All tasks completed!")
    print("=" * 80)
    print(f"\nCorrupted datasets are saved in: {output_dir.absolute()}/")
    if choice == 'all':
        print("  corrupted_data_<gan_type>/ (for each GAN type)")
    else:
        print(f"  corrupted_data_{gan_type}/")
    print("\nEach directory contains:")
    print("  - original/ (original images)")
    print("  - contrast/ (contrast corruption)")
    print("  - gaussian_noise/ (Gaussian noise)")
    print("  - motion_blur/ (motion blur)")
    print("  - pixelate/ (pixelation)")
    print("  - jpeg_compression/ (JPEG artifacts)")
    print("  - fog/ (fog effect)")
    print("\nDataset source:")
    print("  - Original dataset location: dataset/test/<gan_type>/")
    print("  - Corrupted data location: corrupted_data_<gan_type>/")


if __name__ == "__main__":
    main()
