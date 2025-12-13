"""
Quick test for FreqNorm implementation
"""

import torch
from torchvision import transforms
from model.method.freq_norm import create_lgrad_freqnorm
from utils.data.dataset import CorruptedDataset

def test_freqnorm():
    """Quick test to verify FreqNorm works"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Create model
    print("Creating LGrad + FreqNorm model...")
    model = create_lgrad_freqnorm(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        reference_stats_path="/workspace/robust_deepfake_ai/freq_reference_progan.pth",
        rho=0.5,
        device=device,
    )
    model.eval()
    print("Model created!\n")

    # Load small dataset
    print("Loading test data...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CorruptedDataset(
        root="/workspace/robust_deepfake_ai/corrupted_dataset",
        datasets=["corrupted_test_data_progan"],
        corruptions=["gaussian_noise"],  # Test on noise corruption
        transform=transform,
    )
    print(f"Dataset: {len(dataset)} samples\n")

    # Test on a few samples
    print("Testing inference...")
    n_test = 5

    for i in range(n_test):
        img, label, metadata = dataset[i]
        img_batch = img.unsqueeze(0).to(device)

        # Inference
        prob = model.predict_proba(img_batch).item()
        pred = model.predict(img_batch).item()

        label_str = "Fake" if label == 1 else "Real"
        pred_str = "Fake" if pred == 1 else "Real"

        print(f"Sample {i+1}: True={label_str}, Pred={pred_str}, Prob={prob:.3f}")

    print("\n✓ FreqNorm test passed!")


if __name__ == "__main__":
    test_freqnorm()
