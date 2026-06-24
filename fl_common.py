import os
import sys
import time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import random
import numpy as np

BASE_SEED = int(os.environ.get("FL_SEED", 42))
STRICT_DETERMINISM = os.environ.get("STRICT_DETERMINISM", "1") == "1"

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if STRICT_DETERMINISM:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

set_global_seed(BASE_SEED)
print(f"[SEED] FL_SEED={BASE_SEED} | strict={STRICT_DETERMINISM}", flush=True)

# ============================================================
# Path setup for moving this file into: CVMRI/FL/fl_common.py
# ============================================================
FL_DIR = Path(__file__).resolve().parent
CVMRI_ROOT = FL_DIR.parent

# Make imports work when running from CVMRI/FL
if str(CVMRI_ROOT) not in sys.path:
    sys.path.insert(0, str(CVMRI_ROOT))

from models.pGAN.networks import define_G, define_D, GANLoss
from utils.build_df import (
    build_patient_modality_dataframe,
    count_unpaired,
    modality_distribution,
    PairDataset,
)
from trainer.trainer import train_pix2pix, evaluate_and_generate
from preprocessing.cut_edge import cut_edge_and_return_dataset

START_TIME = time.time()

# ============================================================
# Main paths
# Override MRI root at submit time if needed:
#   export MRI_ROOT=/scratch/.../TCGA-GBM_MRI
# ============================================================
mri_root = os.environ.get(
    "MRI_ROOT",
    "/frozen/kustats/GBM_Open_Source/TCGA-GBM/manifest-1627065227338/manifest-7AMtkRMr1024802740904172577/TCGA-GBM_MRI",
)

OUTPUT_ROOT = Path(os.environ.get("FL_OUTPUT_ROOT", FL_DIR / "outputs"))
LOG_ROOT = Path(os.environ.get("FL_LOG_ROOT", FL_DIR / "logs"))

checkpoint_pGAN = str(OUTPUT_ROOT / "checkpoints" / "pGANFL")
file_name_pGAN_single = str(OUTPUT_ROOT / "generated" / "pGANFL")

for p in [
    OUTPUT_ROOT,
    OUTPUT_ROOT / "checkpoints",
    OUTPUT_ROOT / "generated",
    OUTPUT_ROOT / "history",
    OUTPUT_ROOT / "server_states",
    LOG_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)

SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 50))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
CLIENT_BATCH_SIZE = int(os.environ.get("CLIENT_BATCH_SIZE", 8))
CLIENT_LR = float(os.environ.get("CLIENT_LR", 0.0002))
CLIENT_BETAS = (0.5, 0.999)
CLIENT_VAL_RATIO = float(os.environ.get("CLIENT_VAL_RATIO", 0.2))
NUM_WORKERS = int(os.environ.get("FL_NUM_WORKERS", 8))

MODALITIES = [
    "axial_T1", "coronal_T1", "sagittal_T1",
    "axial_T2", "coronal_T2", "sagittal_T2",
    "axial_FLAIR", "coronal_FLAIR", "sagittal_FLAIR",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DATASET_CACHE = None
_CLIENT_LOADER_CACHE = {}
_CLIENT_DATASET_CACHE = {}

TOP_SITE_IDS = os.environ.get("TOP_SITE_IDS", "06,02,76,12").split(",")
NUM_CLIENTS = len(TOP_SITE_IDS)


def lambda_rule(epoch):
    decay_start = 390
    total_epochs = 400
    if epoch < decay_start:
        return 1.0
    return 1.0 - (epoch - decay_start) / max(total_epochs - decay_start, 1)


def get_num_site_clients(direction_name):
    print("[DATA] Using fixed top TCGA sites:", TOP_SITE_IDS, flush=True)
    return len(TOP_SITE_IDS)


def create_models(device, seed: int = None):
    if seed is not None:
        set_global_seed(seed)

    netG = define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
    ).to(device)

    netD = define_D(
        input_nc=2,
        ndf=64,
        netD="basic",
        n_layers_D=3,
        norm="batch",
        init_type="normal",
        init_gain=0.02,
    ).to(device)

    return netG, netD


def create_optimizers_and_schedulers(netG, netD):
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=CLIENT_LR, betas=CLIENT_BETAS)
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=CLIENT_LR, betas=CLIENT_BETAS)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    return optimizer_G, optimizer_D, scheduler_G, scheduler_D


def create_models_and_optimizers(device, seed: int = None):
    netG, netD = create_models(device, seed=seed)
    optimizer_G, optimizer_D, scheduler_G, scheduler_D = create_optimizers_and_schedulers(netG, netD)
    return netG, netD, optimizer_G, optimizer_D, scheduler_G, scheduler_D

criterionGAN = GANLoss("vanilla").to(DEVICE)
criterionL1 = nn.L1Loss()
lambda_L1 = 100.0


def get_combined_parameters(netG, netD):
    combined = OrderedDict()

    for _, (k, v) in enumerate(netG.state_dict().items()):
        combined[f"G::{k}"] = v.detach().cpu().numpy()

    for _, (k, v) in enumerate(netD.state_dict().items()):
        combined[f"D::{k}"] = v.detach().cpu().numpy()

    return list(combined.values())


def set_combined_parameters(netG, netD, parameters):
    g_state = OrderedDict()
    d_state = OrderedDict()

    g_keys = list(netG.state_dict().keys())
    d_keys = list(netD.state_dict().keys())

    g_len = len(g_keys)
    g_params = parameters[:g_len]
    d_params = parameters[g_len:]

    for k, v in zip(g_keys, g_params):
        g_state[k] = torch.tensor(v, dtype=netG.state_dict()[k].dtype)

    for k, v in zip(d_keys, d_params):
        d_state[k] = torch.tensor(v, dtype=netD.state_dict()[k].dtype)

    netG.load_state_dict(g_state, strict=True)
    netD.load_state_dict(d_state, strict=True)


def get_tcga_site_id(Patient):
    return str(Patient).split("-")[1]


def _clean_missing_modalities(df):
    planes = ["axial", "coronal", "sagittal"]
    sequences = ["T1", "T2", "FLAIR"]

    for p in planes:
        for s in sequences:
            slice_col = f"{p}_{s}_slices"
            path_col = f"{p}_{s}_path"
            df.loc[df[slice_col] == 0, path_col] = ""

    return df


def build_datasets():
    global _DATASET_CACHE

    if _DATASET_CACHE is not None:
        print("[DATA] Using cached datasets", flush=True)
        return _DATASET_CACHE

    print("[DATA] Building dataframe and datasets...", flush=True)
    print(f"[DATA] MRI root: {mri_root}", flush=True)
    t0 = time.time()

    df = build_patient_modality_dataframe(MODALITIES, mri_root)
    df["site_id"] = df["Patient"].apply(get_tcga_site_id)
    print("[DATA] TCGA sites:", sorted(df["site_id"].unique()), flush=True)

    df = _clean_missing_modalities(df)

    datasets = {
        "T1_to_T2": PairDataset(df, "axial", "T1", "T2"),
        "T2_to_T1": PairDataset(df, "axial", "T2", "T1"),
        "T1_to_FLAIR": PairDataset(df, "axial", "T1", "FLAIR"),
        "T2_to_FLAIR": PairDataset(df, "axial", "T2", "FLAIR"),
        "FLAIR_to_T1": PairDataset(df, "axial", "FLAIR", "T1"),
        "FLAIR_to_T2": PairDataset(df, "axial", "FLAIR", "T2"),
    }

    for key in list(datasets.keys()):
        datasets[key] = cut_edge_and_return_dataset(
            datasets[key],
            start_percent=0.33,
            end_percent=0.15,
        )
        print(f"[DATA] {key} pairs after edge cut: {len(datasets[key])}", flush=True)

    t1_only, t2_only, flair_only = count_unpaired(df, "axial")
    print("[DATA] Unpaired volumes:", flush=True)
    print(f"[DATA] T1 only: {t1_only}", flush=True)
    print(f"[DATA] T2 only: {t2_only}", flush=True)
    print(f"[DATA] FLAIR only: {flair_only}", flush=True)

    dist = modality_distribution(df, "axial")
    print("[DATA] Modality Distribution (Axial):", flush=True)
    for k, v in dist.items():
        print(f"[DATA] {k}: {v}", flush=True)

    _DATASET_CACHE = datasets
    print(f"[DATA] Dataset build finished in {time.time() - t0:.2f}s", flush=True)
    return _DATASET_CACHE


def get_client_loaders(direction_name, client_id, val_ratio=CLIENT_VAL_RATIO):
    global _CLIENT_LOADER_CACHE, _CLIENT_DATASET_CACHE

    cache_key = (direction_name, client_id)

    if cache_key in _CLIENT_LOADER_CACHE:
        print(f"[CLIENT {client_id}] Using cached RAM loaders", flush=True)
        return _CLIENT_LOADER_CACHE[cache_key]

    print(f"[CLIENT {client_id}] Building site-based train/val loaders once...", flush=True)

    if client_id >= len(TOP_SITE_IDS):
        raise ValueError(
            f"Invalid client_id={client_id}. Available: 0-{len(TOP_SITE_IDS)-1}"
        )

    site_id = TOP_SITE_IDS[client_id]
    print(f"[CLIENT {client_id}] Assigned TCGA site={site_id}", flush=True)

    df = build_patient_modality_dataframe(MODALITIES, mri_root)
    df["site_id"] = df["Patient"].apply(get_tcga_site_id)

    site_df = df[df["site_id"] == site_id].reset_index(drop=True)

    print(
        f"[CLIENT {client_id}] site={site_id} patient rows={len(site_df)}",
        flush=True,
    )

    site_df = _clean_missing_modalities(site_df)

    src, tgt = direction_name.split("_to_")
    subset = PairDataset(site_df, "axial", src, tgt)

    print(f"[CLIENT {client_id}] before edge cut samples={len(subset)}", flush=True)

    subset = cut_edge_and_return_dataset(
        subset,
        start_percent=0.33,
        end_percent=0.15,
    )

    print(f"[CLIENT {client_id}] after edge cut samples={len(subset)}", flush=True)

    _CLIENT_DATASET_CACHE[cache_key] = subset

    subset_len = len(subset)
    val_len = max(1, int(val_ratio * subset_len))
    train_len = subset_len - val_len

    if train_len <= 0:
        raise ValueError(
            f"Client {client_id}, TCGA site {site_id}, has too few samples: {subset_len}"
        )

    split_seed = BASE_SEED + int(client_id) * 1000
    generator = torch.Generator().manual_seed(split_seed)

    train_subset, val_subset = random_split(
        subset,
        [train_len, val_len],
        generator=generator,
    )

    loader_kwargs = dict(
        batch_size=CLIENT_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    if NUM_WORKERS > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=torch.Generator().manual_seed(BASE_SEED + int(client_id) * 1000 + 1),
        **loader_kwargs
    )
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

    print(
        f"[CLIENT {client_id}] site={site_id} | direction={direction_name} | "
        f"train_samples={len(train_subset)} | val_samples={len(val_subset)}",
        flush=True,
    )

    _CLIENT_LOADER_CACHE[cache_key] = (train_loader, val_loader)
    return train_loader, val_loader


def get_full_loader(direction_name):
    print(f"[SERVER] Building full loader for {direction_name}...", flush=True)
    datasets = build_datasets()
    loader = DataLoader(
        datasets[direction_name],
        batch_size=CLIENT_BATCH_SIZE,
        shuffle=False,
        num_workers=max(0, min(NUM_WORKERS, 4)),
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[SERVER] Full loader ready for {direction_name}", flush=True)
    return loader


def _extract_batch_tensors(batch):
    if isinstance(batch, dict):
        real_A = batch["A"].to(DEVICE)
        real_B = batch["B"].to(DEVICE)
        return real_A, real_B

    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        real_A = batch[0].to(DEVICE)
        real_B = batch[1].to(DEVICE)
        return real_A, real_B

    raise ValueError("Unsupported batch format. Expected dict with A/B or tuple/list.")


def _simple_ssim_torch(x, y, data_range=2.0, c1=0.01 ** 2, c2=0.03 ** 2):
    """
    Fast validation SSIM approximation.
    Assumes pix2pix tensors are roughly normalized to [-1, 1], so data_range=2.
    Returns batch mean SSIM.
    """
    c1 = (c1 * data_range) ** 2
    c2 = (c2 * data_range) ** 2

    mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)

    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return ssim_map.mean()


def validate_pix2pix_metrics(netG, netD, val_loader, device):
    netG.eval()
    netD.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            real_A, real_B = _extract_batch_tensors(batch)

            fake_B = netG(real_A)

            pred_fake = netD(torch.cat((real_A, fake_B), 1))
            loss_g_gan = criterionGAN(pred_fake, True)
            l1 = criterionL1(fake_B, real_B)
            loss_g = loss_g_gan + l1 * lambda_L1

            mse = F.mse_loss(fake_B, real_B).clamp_min(1e-12)
            psnr = 20.0 * torch.log10(torch.tensor(2.0, device=device)) - 10.0 * torch.log10(mse)
            ssim = _simple_ssim_torch(fake_B, real_B)

            total_loss += float(loss_g.item())
            total_l1 += float(l1.item())
            total_ssim += float(ssim.item())
            total_psnr += float(psnr.item())
            total_batches += 1

    denom = max(total_batches, 1)
    return {
        "val_loss": total_loss / denom,
        "val_l1": total_l1 / denom,
        "val_ssim": total_ssim / denom,
        "val_psnr": total_psnr / denom,
    }


def validate_pix2pix_loss(netG, netD, val_loader, device):
    return validate_pix2pix_metrics(netG, netD, val_loader, device)["val_loss"]


def train_local_client(direction_name, client_id, parameters, local_epochs):

    client_seed = BASE_SEED + int(client_id) * 1000
    set_global_seed(client_seed)
    t0 = time.time()

    train_loader, val_loader = get_client_loaders(direction_name, client_id)

    print(f"[CLIENT {client_id}] Creating local models...", flush=True)
    (
        local_netG,
        local_netD,
        local_optimizer_G,
        local_optimizer_D,
        local_scheduler_G,
        local_scheduler_D,
    ) = create_models_and_optimizers(DEVICE, seed=client_seed)
    print(f"[CLIENT {client_id}] Local models created", flush=True)

    print(f"[CLIENT {client_id}] Loading global parameters...", flush=True)
    set_combined_parameters(local_netG, local_netD, parameters)
    print(f"[CLIENT {client_id}] Global parameters loaded", flush=True)

    print(
        f"[CLIENT {client_id}] Local training started | direction={direction_name} | epochs={local_epochs}",
        flush=True,
    )

    train_pix2pix(
        loader=train_loader,
        direction_name=f"{direction_name}_client_{client_id}",
        netG=local_netG,
        netD=local_netD,
        optimizer_G=local_optimizer_G,
        optimizer_D=local_optimizer_D,
        scheduler_G=local_scheduler_G,
        scheduler_D=local_scheduler_D,
        criterionGAN=criterionGAN,
        criterionL1=criterionL1,
        lambda_L1=lambda_L1,
        device=DEVICE,
        epochs=local_epochs,
        use_pretrained=False,
        save_every=SAVE_EVERY,
        checkpoint=checkpoint_pGAN,
    )

    print(f"[CLIENT {client_id}] Local training finished", flush=True)

    metrics = validate_pix2pix_metrics(local_netG, local_netD, val_loader, DEVICE)
    updated_parameters = get_combined_parameters(local_netG, local_netD)
    elapsed = time.time() - t0

    print(
        f"[CLIENT {client_id}] train_local_client done | "
        f"train_samples={len(train_loader.dataset)} | "
        f"val_samples={len(val_loader.dataset)} | "
        f"val_loss={metrics['val_loss']:.6f} | "
        f"val_ssim={metrics['val_ssim']:.6f} | "
        f"val_psnr={metrics['val_psnr']:.6f} | "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )

    metrics["local_steps"] = int(len(train_loader) * local_epochs)
    return updated_parameters, len(train_loader.dataset), metrics


def evaluate_global(direction_name, parameters):
    print(f"[SERVER] Starting global evaluation for {direction_name}", flush=True)

    global_netG, global_netD = create_models(DEVICE)
    set_combined_parameters(global_netG, global_netD, parameters)

    eval_optimizer_G, eval_optimizer_D, eval_scheduler_G, eval_scheduler_D = \
        create_optimizers_and_schedulers(global_netG, global_netD)

    loader = get_full_loader(direction_name)

    print(f"[SERVER] Building eval loader for {direction_name}", flush=True)

    """test_loader = train_pix2pix(
        loader=loader,
        direction_name=f"{direction_name}_global_eval_loader",
        netG=global_netG,
        netD=global_netD,
        optimizer_G=eval_optimizer_G,
        optimizer_D=eval_optimizer_D,
        scheduler_G=eval_scheduler_G,
        scheduler_D=eval_scheduler_D,
        criterionGAN=criterionGAN,
        criterionL1=criterionL1,
        lambda_L1=lambda_L1,
        device=DEVICE,
        epochs=1,
        use_pretrained=False,
        save_every=SAVE_EVERY,
        checkpoint=checkpoint_pGAN,
    )"""
    test_loader = loader

    ssim, psnr = evaluate_and_generate(
        test_loader=test_loader,
        file_name=file_name_pGAN_single,
        direction_name=direction_name,
        netG=global_netG,
        device=DEVICE,
        max_slices=30,
    )

    print(f"[SERVER] Eval done | SSIM={ssim:.6f} | PSNR={psnr:.6f}", flush=True)
    return {"SSIM": float(ssim), "PSNR": float(psnr)}
