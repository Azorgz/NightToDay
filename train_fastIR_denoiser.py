import torch
import torch.nn as nn
import torch.optim as optim
from ImagesCameras import ImageTensor
from ImagesCameras.Metrics import SSIM
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import math
import os
from PIL import Image
from tqdm import tqdm

from NightToday.Fusion import FastIRDenoiser
from NightToday.losses import TVLoss, ContrastiveLoss


# Assuming you saved the previous code in a file named `models.py`
# from models import FastIRDenoiser

class SyntheticNoiseDataset(Dataset):
    """
    Takes clean IR images and applies synthetic noise on the fly.
    This creates perfect paired data for training the denoiser.
    """

    def __init__(self, image_dir, image_size=256, noise_level=0.002):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                            f.endswith(('.png', '.jpg', '.jpeg'))]
        self.noise_level = noise_level

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=1),  # Ensure 1-channel IR
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        clean_img = self.transform(Image.open(img_path))

        # Inject Gaussian Noise
        clean_img = clean_img * (0.5 + torch.rand(1) * 0.5)  # Randomly scale down to simulate varying conditions
        if torch.rand(1) < 0.5:
            noise_scale = 100.0
            # 1. Multiply by scale to get the Poisson rate (lambda)
            rate = clean_img * noise_scale
            # 2. Sample from the Poisson distribution
            noisy_img = torch.poisson(rate)
            # 3. Divide by the scale to bring it back to the [0, 1] range
            noisy_img = noisy_img / noise_scale
            noisy_img = noisy_img * self.noise_level + clean_img * (1 - self.noise_level)
        else:
            noise = torch.randn_like(clean_img) * self.noise_level * torch.rand(1)  # Random noise level for each image
            noisy_img = clean_img + noise
        clean_img = (clean_img - clean_img.min()) / (clean_img.max() - clean_img.min())  # Normalize to [0, 1]

        # Clamp to valid image range [0, 1]
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        return noisy_img * 2 - 1, clean_img * 2 - 1


def calculate_psnr(mse, max_pixel_val=1.0):
    """ Calculates Peak Signal-to-Noise Ratio for evaluation. """
    if mse == 0: return float('inf')
    return 20 * math.log10(max_pixel_val / math.sqrt(mse))


def train_denoiser(data_dir, epochs=50, batch_size=16, lr=1e-4, device='cuda'):
    print("Initializing standalone denoiser training...")

    # 1. Setup Data
    dataset = SyntheticNoiseDataset(image_dir=data_dir, noise_level=0.05)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2. Initialize Model (Imported from your modules)
    # model = FastIRDenoiser(in_c=1, base_c=32, num_blocks=2).to(device)

    # For this script's completeness, assuming `model` is instantiated here
    model = FastIRDenoiser().to(device)

    # resume from checkpoint if exists
    epoch = 10
    checkpoint_path = f"checkpoints/fast_ir_denoiser_epoch.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"--> Resumed from checkpoint: {checkpoint_path}")

    # 3. Loss & Optimizer
    # L1 Loss is strongly preferred over MSE (L2) for image restoration to avoid blurring
    criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()  # Used strictly for calculating PSNR
    ssim_criterion = SSIM(device)  # SSIM expects input in range [-1, 1]
    TV_criterion = TVLoss()
    contrast_criterion = ContrastiveLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    bar = tqdm(range(epochs * len(dataloader)), desc="Training Denoiser")
    screen = None  # For real-time visualization of training progress
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        epoch_tv = 0.0
        epoch_contrast = 0.0

        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(dataloader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            # Forward pass
            optimizer.zero_grad()
            restored_imgs = model(noisy_imgs)

            # Loss calculation
            loss = criterion(restored_imgs, clean_imgs)
            ssim = ssim_criterion(restored_imgs.mean(1, keepdim=True), clean_imgs.mean(1, keepdim=True)).mean()
            tv = TV_criterion(restored_imgs)
            contrast = contrast_criterion(restored_imgs)
            loss += tv * 0.1 + contrast * 0.5 # TV loss helps to reduce noise
            loss -= ssim * 0.05

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate metrics (detached from graph)
            with torch.no_grad():
                mse = mse_criterion(restored_imgs, clean_imgs).item()
                epoch_psnr += calculate_psnr(mse)
                epoch_ssim += ssim.item()
                epoch_tv += tv.item()
                epoch_contrast += contrast.item()
            bar.update(1)
            bar.set_postfix({"Epoch": epoch + 1, "Batch Loss": loss.item(),
                             "PSNR": epoch_psnr / (batch_idx + 1),
                             "SSIM": epoch_ssim / (batch_idx + 1),
                             "TV": epoch_tv / (batch_idx + 1),
                             "Contrast": epoch_contrast / (batch_idx + 1)})
            if batch_idx % 10 == 0:
                compose = ImageTensor(clean_imgs[0]).hstack(ImageTensor(noisy_imgs[0])).hstack(ImageTensor(restored_imgs[0]))
                if screen is None:
                    screen = (compose*0.5+0.5).show('training ongoing', opencv=True, asyncr=True)
                else:
                    screen.update(compose*0.5+0.5)


        # Adjust learning rate
        scheduler.step()

        # Logging
        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)

        print(f"Epoch [{epoch + 1}/{epochs}] | L1 Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

        # Save checkpoints periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/fast_ir_denoiser_epoch_{epoch + 1}.pth")
            print(f"--> Saved checkpoint: fast_ir_denoiser_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    # Point this to a folder containing your cleanest IR images
    train_denoiser(data_dir="/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/trainB", epochs=50)