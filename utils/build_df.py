import os
import time
import platform
from typing import Optional

import numpy as np
import pandas as pd
import pydicom
import psutil
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from torch.utils.data import Dataset
from models.pGAN.networks import define_G

MODALITIES = [
    "axial_T1", "coronal_T1", "sagittal_T1",
    "axial_T2", "coronal_T2", "sagittal_T2",
    "axial_FLAIR", "coronal_FLAIR", "sagittal_FLAIR",
]

def normalize_modality(name: str) -> Optional[str]:
    name = name.lower()

    if "ax" in name and "t1" in name:
        return "axial_T1"
    if "cor" in name and "t1" in name:
        return "coronal_T1"
    if "sag" in name and "t1" in name:
        return "sagittal_T1"

    if "ax" in name and "t2" in name:
        return "axial_T2"
    if "cor" in name and "t2" in name:
        return "coronal_T2"
    if "sag" in name and "t2" in name:
        return "sagittal_T2"

    if "ax" in name and "flair" in name:
        return "axial_FLAIR"
    if "cor" in name and "flair" in name:
        return "coronal_FLAIR"
    if "sag" in name and "flair" in name:
        return "sagittal_FLAIR"

    return None

def count_dicom_slices(series_dir: str) -> int:
    try:
        return sum(
            1 for f in os.listdir(series_dir)
            if f.lower().endswith(".dcm")
        )
    except Exception:
        return 0

def build_patient_modality_dataframe(MODALITIES, mri_root: str) -> pd.DataFrame:
    records = []

    for patient in sorted(os.listdir(mri_root)):
        patient_dir = os.path.join(mri_root, patient)
        if not os.path.isdir(patient_dir) or patient.startswith("."):
            continue

        sessions = sorted([
            s for s in os.listdir(patient_dir)
            if os.path.isdir(os.path.join(patient_dir, s))
            and not s.startswith(".")
        ])
        if not sessions:
            continue

        session_dir = os.path.join(patient_dir, sessions[0])

        row = {"Patient": patient}

        # initialize all modalities
        for mod in MODALITIES:
            row[f"{mod}_slices"] = 0
            row[f"{mod}_path"] = ""

        for series in os.listdir(session_dir):
            series_path = os.path.join(session_dir, series)
            if not os.path.isdir(series_path):
                continue

            mod = normalize_modality(series)
            if mod is None:
                continue

            n_slices = count_dicom_slices(series_path)

            row[f"{mod}_slices"] = n_slices
            row[f"{mod}_path"] = series_path

        records.append(row)

    return pd.DataFrame(records)


# =========================================================
# System Info Logger
# =========================================================
def log_system_info(START_TIME):
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    # ---- OS ----
    print(f"OS                : {platform.system()} {platform.release()}")
    print(f"Machine           : {platform.machine()}")

    # ---- CPU ----
    print(f"CPU               : {platform.processor()}")
    print(f"Physical Cores    : {psutil.cpu_count(logical=False)}")
    print(f"Logical Cores     : {psutil.cpu_count(logical=True)}")

    # ---- RAM ----
    ram = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM         : {ram:.2f} GB")

    # ---- PyTorch ----
    print(f"PyTorch Version   : {torch.__version__}")

    # ---- CUDA / GPU ----
    if torch.cuda.is_available():
        print("CUDA Available    : Yes")
        print(f"CUDA Version      : {torch.version.cuda}")
        print(f"GPU Count         : {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i} Name      : {gpu_name}")
            print(f"GPU {i} VRAM      : {total_mem:.2f} GB")
    else:
        print("CUDA Available    : No")

        # ---- Runtime ----
    elapsed = time.time() - START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"Running Time      : {hours}h {minutes}m {seconds}s")
    
    print("=" * 60)      


def load_dicom_series_raw(series_dir):
    """Load a DICOM folder into a raw stack volume (X,Y,Slices) and spacing (sx,sy,sz)."""
    if series_dir is None or series_dir == "" or (not os.path.isdir(series_dir)):
        return None, None

    files = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.lower().endswith(".dcm")
    ]
    if not files:
        return None, None

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=False)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        return None, None

    # Sort slices robustly
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except Exception:
            pass

    vol = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float32)

    # Spacing (best-effort)
    try:
        sx = float(slices[0].PixelSpacing[0])
        sy = float(slices[0].PixelSpacing[1])
    except Exception:
        sx, sy = 1.0, 1.0

    try:
        sz = float(getattr(slices[0], "SliceThickness", 1.0))
    except Exception:
        sz = 1.0

    return vol, (sx, sy, sz)

class PairDataset(Dataset):
    def __init__(self, df, plane, src_mod, tgt_mod):
        self.samples = []

        for _, row in df.iterrows():
            src_path = row[f"{plane}_{src_mod}_path"]
            tgt_path = row[f"{plane}_{tgt_mod}_path"]

            if src_path == "" or tgt_path == "":
                continue

            src_vol, _ = load_dicom_series_raw(src_path)
            tgt_vol, _ = load_dicom_series_raw(tgt_path)

            if src_vol is None or tgt_vol is None:
                continue

            depth = min(src_vol.shape[2], tgt_vol.shape[2])

            for i in range(depth):
                self.samples.append({
                    "src_vol": src_vol,
                    "tgt_vol": tgt_vol,
                    "slice_idx": i
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        src = item["src_vol"][:, :, item["slice_idx"]]
        tgt = item["tgt_vol"][:, :, item["slice_idx"]]

        # ---------- normalize while numpy ----------
        def normalize(img):
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img * 2 - 1
            return img

        src = normalize(src)
        tgt = normalize(tgt)

        # ---------- convert to torch ----------
        src = torch.from_numpy(src).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)

        # ---------- resize ----------
        src = F.interpolate(src.unsqueeze(0), size=(256,256),
                            mode="bilinear", align_corners=False).squeeze(0)
        tgt = F.interpolate(tgt.unsqueeze(0), size=(256,256),
                            mode="bilinear", align_corners=False).squeeze(0)

        return src, tgt

def count_unpaired(df, plane):
    t1_only = 0
    t2_only = 0
    flair_only = 0

    for _, row in df.iterrows():
        has_t1 = row[f"{plane}_T1_slices"] > 0
        has_t2 = row[f"{plane}_T2_slices"] > 0
        has_flair = row[f"{plane}_FLAIR_slices"] > 0

        if has_t1 and not has_t2 and not has_flair:
            t1_only += 1
        if has_t2 and not has_t1 and not has_flair:
            t2_only += 1
        if has_flair and not has_t1 and not has_t2:
            flair_only += 1

    return t1_only, t2_only, flair_only

def modality_distribution(df, plane):
    counts = {
        "T1": 0,
        "T2": 0,
        "FLAIR": 0,
        "T1+T2": 0,
        "T1+FLAIR": 0,
        "T2+FLAIR": 0,
        "T1+T2+FLAIR": 0,
    }

    for _, row in df.iterrows():
        has_t1 = row[f"{plane}_T1_slices"] > 0
        has_t2 = row[f"{plane}_T2_slices"] > 0
        has_flair = row[f"{plane}_FLAIR_slices"] > 0

        if has_t1: counts["T1"] += 1
        if has_t2: counts["T2"] += 1
        if has_flair: counts["FLAIR"] += 1

        if has_t1 and has_t2:
            counts["T1+T2"] += 1
        if has_t1 and has_flair:
            counts["T1+FLAIR"] += 1
        if has_t2 and has_flair:
            counts["T2+FLAIR"] += 1
        if has_t1 and has_t2 and has_flair:
            counts["T1+T2+FLAIR"] += 1

    return counts

def load_generator(checkpoint_path, device):

    netG = define_G(
    input_nc=1,
    output_nc=1,
    ngf=64,
    netG="unet_256",     
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    )
 
    checkpoint = torch.load(checkpoint_path, map_location=device)

    netG.load_state_dict(checkpoint["netG_state_dict"])

    netG.to(device)
    netG.eval()

    print(f"Loaded generator from epoch {checkpoint['epoch']}")

    return netG  

def generate_volume(vol, netG, save_dir, device):

    os.makedirs(save_dir, exist_ok=True)

    for i in range(vol.shape[2]):

        img = vol[:, :, i].astype(np.float32)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 2 - 1

        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

        tensor = F.interpolate(
            tensor,
            size=(256,256),
            mode="bilinear",
            align_corners=False
        )

        tensor = tensor.to(device)

        with torch.no_grad():
            fake = netG(tensor)

        fake = (fake.squeeze().cpu() + 1) / 2

        vutils.save_image(
            fake,
            f"{save_dir}/slice_{i:04d}.png"
        )


def generate_missing_modalities(df, device, checkpoint_root, output_root):

    os.makedirs(output_root, exist_ok=True)

    print("Checkpoint root:", checkpoint_root)
    for d in os.listdir(checkpoint_root):
        print(d, "->", os.listdir(os.path.join(checkpoint_root, d)))

    generators = {
        "T1_to_T2": load_generator(
            f"{checkpoint_root}/T1_to_T2/latest_checkpoint.pth", device
        ),
        "T1_to_FLAIR": load_generator(
            f"{checkpoint_root}/T1_to_FLAIR/latest_checkpoint.pth", device
        ),
        "T2_to_T1": load_generator(
            f"{checkpoint_root}/T2_to_T1/latest_checkpoint.pth", device
        ),
        "T2_to_FLAIR": load_generator(
            f"{checkpoint_root}/T2_to_FLAIR/latest_checkpoint.pth", device
        ),
    }

    for _, row in df.iterrows():

        patient_id = row["Patient"]

        patient_dir = os.path.join(output_root, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        has_T1 = row["axial_T1_slices"] > 0
        has_T2 = row["axial_T2_slices"] > 0
        has_FLAIR = row["axial_FLAIR_slices"] > 0

        T1_path = row["axial_T1_path"]
        T2_path = row["axial_T2_path"]
        FLAIR_path = row["axial_FLAIR_path"]

        # -------------------------------------------------
        # Load available volumes
        # -------------------------------------------------

        T1_vol, _ = load_dicom_series_raw(T1_path) if has_T1 else (None, None)
        T2_vol, _ = load_dicom_series_raw(T2_path) if has_T2 else (None, None)

        # -------------------------------------------------
        # Generate T2 if missing
        # -------------------------------------------------

        if not has_T2 and has_T1:

            print(f"Generating T2 for patient {patient_id}")

            netG = generators["T1_to_T2"]

            save_dir = os.path.join(patient_dir, "T2")
            os.makedirs(save_dir, exist_ok=True)

            generate_volume(T1_vol, netG, save_dir, device)

        # -------------------------------------------------
        # Generate FLAIR if missing
        # -------------------------------------------------

        if not has_FLAIR:

            if has_T1:

                print(f"Generating FLAIR from T1 for {patient_id}")

                netG = generators["T1_to_FLAIR"]
                save_dir = os.path.join(patient_dir, "FLAIR")

                generate_volume(T1_vol, netG, save_dir, device)

            elif has_T2:

                print(f"Generating FLAIR from T2 for {patient_id}")

                netG = generators["T2_to_FLAIR"]
                save_dir = os.path.join(patient_dir, "FLAIR")

                generate_volume(T2_vol, netG, save_dir, device)

        # -------------------------------------------------
        # Generate T1 if missing
        # -------------------------------------------------

        if not has_T1 and has_T2:

            print(f"Generating T1 for patient {patient_id}")

            netG = generators["T2_to_T1"]

            save_dir = os.path.join(patient_dir, "T1")
            os.makedirs(save_dir, exist_ok=True)

            generate_volume(T2_vol, netG, save_dir, device)


def count_generated_slices(folder):
    if not os.path.isdir(folder):
        return 0

    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])

def add_synthesized_modalities(df, generated_root):

    new_df = df.copy()

    for idx, row in new_df.iterrows():

        patient = row["Patient"]
        patient_dir = os.path.join(generated_root, patient)

        if not os.path.isdir(patient_dir):
            continue

        for modality in ["T1", "T2", "FLAIR"]:

            synth_dir = os.path.join(patient_dir, modality)

            n_slices = count_generated_slices(synth_dir)

            if n_slices > 0:

                slice_col = f"axial_{modality}_slices"
                path_col = f"axial_{modality}_path"

                # update dataframe
                new_df.at[idx, slice_col] = n_slices
                new_df.at[idx, path_col] = synth_dir

                print(f"Synthetic {modality} added for {patient} ({n_slices} slices)")

    return new_df    

def analyze_modalities(df):

    stats = {
        "raw": {"T1":0,"T2":0,"FLAIR":0},
        "synth":{"T1":0,"T2":0,"FLAIR":0}
    }

    slice_stats = {
        "raw": {"T1":0,"T2":0,"FLAIR":0},
        "synth":{"T1":0,"T2":0,"FLAIR":0}
    }

    for _, row in df.iterrows():

        for mod in ["T1","T2","FLAIR"]:

            path = row[f"axial_{mod}_path"]
            slices = row[f"axial_{mod}_slices"]

            if path == "" or slices == 0:
                continue

            if "patients_generated_mri" in path:
                stats["synth"][mod] += 1
                slice_stats["synth"][mod] += slices
            else:
                stats["raw"][mod] += 1
                slice_stats["raw"][mod] += slices

    return stats, slice_stats

def count_triplets_patient(df):

    raw_triplets = 0
    synth_triplets = 0

    for _, row in df.iterrows():

        paths = {
            "T1": row["axial_T1_path"],
            "T2": row["axial_T2_path"],
            "FLAIR": row["axial_FLAIR_path"]
        }

        if all(paths[m] != "" for m in paths):

            if all("patients_generated_mri" not in str(paths[m]) for m in paths):    
                raw_triplets += 1
            else:
                synth_triplets += 1

    return raw_triplets, synth_triplets   

def count_triplets_slices(df):

    raw_slices = 0
    synth_slices = 0

    for _, row in df.iterrows():

        t1 = row["axial_T1_slices"]
        t2 = row["axial_T2_slices"]
        fl = row["axial_FLAIR_slices"]

        if t1>0 and t2>0 and fl>0:

            min_slices = min(t1,t2,fl)

            paths = [
                row["axial_T1_path"],
                row["axial_T2_path"],
                row["axial_FLAIR_path"]
            ]

            if all("patients_generated_mri" not in p for p in paths):
                raw_slices += min_slices
            else:
                synth_slices += min_slices

    return raw_slices, synth_slices   