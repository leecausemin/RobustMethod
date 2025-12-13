"""
Organize corrupted dataset into real/fake folders based on filename
"""
import os
from pathlib import Path
from tqdm import tqdm
import shutil

def organize_dataset(root_dir):
    """
    Organize corrupted datasets into real/fake structure based on filename

    Args:
        root_dir: Root directory containing corrupted_test_data_* folders
    """
    root = Path(root_dir)

    # Find all corrupted_test_data_* directories
    dataset_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith('corrupted_test_data_')])

    print(f"Found {len(dataset_dirs)} dataset directories")

    for dataset_dir in dataset_dirs:
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_dir.name}")
        print(f"{'='*70}")

        # Find all corruption type directories
        corruption_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

        for corruption_dir in corruption_dirs:
            print(f"\n  Processing {corruption_dir.name}...")

            # Create real/fake subdirectories
            real_dir = corruption_dir / 'real'
            fake_dir = corruption_dir / 'fake'
            real_dir.mkdir(exist_ok=True)
            fake_dir.mkdir(exist_ok=True)

            # Get all image files in the corruption directory (not in subdirectories)
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                image_files.extend(corruption_dir.glob(f'*{ext}'))

            # Filter out files that are already in real/fake subdirectories
            image_files = [f for f in image_files if f.parent == corruption_dir]

            if not image_files:
                print(f"    No images to organize (already organized or empty)")
                continue

            real_count = 0
            fake_count = 0
            unknown_count = 0

            for image_file in tqdm(image_files, desc=f"    Organizing", leave=False):
                filename = image_file.name.lower()

                # Determine if real or fake based on filename
                if '_0_real_' in filename or 'real' in filename:
                    dest = real_dir / image_file.name
                    shutil.move(str(image_file), str(dest))
                    real_count += 1
                elif '_1_fake_' in filename or 'fake' in filename or 'synthetic' in filename:
                    dest = fake_dir / image_file.name
                    shutil.move(str(image_file), str(dest))
                    fake_count += 1
                else:
                    # If can't determine, put in fake by default (CNNDetection is mostly fake)
                    dest = fake_dir / image_file.name
                    shutil.move(str(image_file), str(dest))
                    unknown_count += 1

            print(f"    ✓ Real: {real_count}, Fake: {fake_count}, Unknown (→fake): {unknown_count}")

    print(f"\n{'='*70}")
    print("Organization complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    root_dir = '/workspace/robust_deepfake_ai/corrupted_dataset'
    print(f"Organizing datasets in: {root_dir}")
    organize_dataset(root_dir)
