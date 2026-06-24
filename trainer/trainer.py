import os
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, random_split
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from .utils import save_json, save_csv, get_timestamp


def train_pix2pix(loader,
                  direction_name,
                  netG,
                  netD,
                  optimizer_G,
                  optimizer_D,
                  scheduler_G,
                  scheduler_D,
                  criterionGAN,
                  criterionL1,
                  lambda_L1,
                  device,
                  epochs=400,
                  use_pretrained=False,
                  save_every=50,
                  checkpoint='checkpoints2'      #saved_models/pGAN/{checkpoint_dir}/{direction_name}/latest_checkpoint.pth
                  ):

    print(f"\nTraining {direction_name}")

    train_size = int(0.8 * len(loader.dataset))
    test_size = len(loader.dataset) - train_size
    #train_set, test_set = random_split(loader.dataset, [train_size, test_size])
    seed = int(os.environ.get("FL_SEED", 42))
    train_set, test_set = random_split(
        loader.dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1)
    )
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    import psutil
    print("RAM usage:", psutil.virtual_memory().percent)

    # --------------------------------------------------
    # Checkpoint Setup
    # --------------------------------------------------
    checkpoint_dir = f"{checkpoint}/{direction_name}"
    os.makedirs(f"saved_models/pGAN/{checkpoint_dir}", exist_ok=True)
    checkpoint_path = f"saved_models/pGAN/{checkpoint_dir}/latest_checkpoint.pth"

    start_epoch = 0

    # --------------------------------------------------
    # Load pretrained checkpoint if requested
    # --------------------------------------------------
    if use_pretrained and os.path.exists(checkpoint_path):

        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        netG.load_state_dict(checkpoint["netG_state_dict"])
        netD.load_state_dict(checkpoint["netD_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")

    else:
        print("Training from scratch")

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    for epoch in range(start_epoch, epochs):

        netG.train()
        netD.train()

        total_g, total_d = 0, 0

        for real_A, real_B in train_loader:
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            g_loss, d_loss = train_step(
                                real_A, real_B,
                                netG, netD,
                                optimizer_G, optimizer_D,
                                criterionGAN, criterionL1,
                                lambda_L1
                            )

            total_g += g_loss
            total_d += d_loss

        print(f"{direction_name} | Epoch {epoch+1}/{epochs} "
              f"| D: {total_d/len(train_loader):.4f} "
              f"G: {total_g/len(train_loader):.4f}")

        scheduler_G.step()
        scheduler_D.step()
        print("Current LR:", scheduler_G.get_last_lr()[0])
        # --------------------------------------------------
        # Save every N epochs
        # --------------------------------------------------
        if (epoch + 1) % save_every == 0:

            checkpoint = {
                "epoch": epoch,
                "netG_state_dict": netG.state_dict(),
                "netD_state_dict": netD.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
            }

            torch.save(checkpoint, checkpoint_path)

            print(f"Checkpoint saved at epoch {epoch+1}")

    # --------------------------------------------------
    # Final Save (Optional but Safe)
    # --------------------------------------------------
    checkpoint = {
        "epoch": epochs - 1,
        "netG_state_dict": netG.state_dict(),
        "netD_state_dict": netD.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)

    print("Final model saved.")

    return test_loader

def evaluate_and_generate(test_loader, file_name, direction_name, netG, device, max_slices=None):
    netG.eval()

    # -------- Basic Metrics --------
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)

    save_dir = f"patients_generated_mri/pGAN_generated_single/{file_name}/{direction_name}/single_patient"
    os.makedirs(save_dir, exist_ok=True)

    slice_counter = 0
    first_patient_detected = False

    with torch.no_grad():

        for real_A, real_B in test_loader:

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B = netG(real_A)

            # ----- Metrics -----
            ssim_metric.update(fake_B, real_B)
            psnr_metric.update(fake_B, real_B)

            # ----- Save only first patient slices in order -----
            for i in range(fake_B.size(0)):

                if max_slices is not None and slice_counter >= max_slices:
                    break

                real_T1 = (real_A[i] + 1) / 2
                fake_T2 = (fake_B[i] + 1) / 2
                real_T2 = (real_B[i] + 1) / 2

                vutils.save_image(
                    real_T1,
                    f"{save_dir}/slice_{slice_counter:03d}_real_T1.png"
                )

                vutils.save_image(
                    fake_T2,
                    f"{save_dir}/slice_{slice_counter:03d}_fake_T2.png"
                )

                vutils.save_image(
                    real_T2,
                    f"{save_dir}/slice_{slice_counter:03d}_real_T2.png"
                )

                slice_counter += 1

            # Stop after finishing first patient volume
            if max_slices is not None and slice_counter >= max_slices:
                break

    # -------- Compute Metrics --------
    ssim = ssim_metric.compute().item()
    psnr = psnr_metric.compute().item()

    print(f"\n{direction_name} Evaluation")
    print(f"SSIM: {ssim:.4f}")
    print(f"PSNR: {psnr:.4f}")

    return ssim, psnr

def train_step(real_A, real_B,
               netG, netD,
               optimizer_G, optimizer_D,
               criterionGAN, criterionL1,
               lambda_L1):

    ###################################
    # 1. Forward
    ###################################
    fake_B = netG(real_A)

    ###################################
    # 2. Train Discriminator
    ###################################
    optimizer_D.zero_grad()

    # Fake
    fake_AB = torch.cat((real_A, fake_B.detach()), 1)
    pred_fake = netD(fake_AB)
    loss_D_fake = criterionGAN(pred_fake, False)

    # Real
    real_AB = torch.cat((real_A, real_B), 1)
    pred_real = netD(real_AB)
    loss_D_real = criterionGAN(pred_real, True)

    loss_D = (loss_D_fake + loss_D_real) * 0.5
    loss_D.backward()
    optimizer_D.step()

    ###################################
    # 3. Train Generator
    ###################################
    optimizer_G.zero_grad()

    fake_AB = torch.cat((real_A, fake_B), 1)
    pred_fake = netD(fake_AB)

    loss_G_GAN = criterionGAN(pred_fake, True)
    loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1

    loss_G = loss_G_GAN + loss_G_L1
    loss_G.backward()
    optimizer_G.step()

    return loss_G.item(), loss_D.item()   
