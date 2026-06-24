import os

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

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


def cut_edge_and_return_dataset(
    dataset,
    start_percent=0.2,
    end_percent=0.1
):
    """
    Cuts edge slices per patient and returns a new PairDataset-like object.
    """

    # --------------------------------------
    # Step 1: Group by patient
    # --------------------------------------
    patients = []
    current_patient = []
    prev_vol_id = None

    for sample in dataset.samples:
        vol_id = id(sample["src_vol"])

        if prev_vol_id is None:
            current_patient.append(sample)

        elif vol_id == prev_vol_id:
            current_patient.append(sample)

        else:
            patients.append(current_patient)
            current_patient = [sample]

        prev_vol_id = vol_id

    if current_patient:
        patients.append(current_patient)

    #print(f"Detected {len(patients)} patients")

    # --------------------------------------
    # Step 2: Apply cutting
    # --------------------------------------
    new_samples = []

    for idx, patient_slices in enumerate(patients):

        original_n = len(patient_slices)

        cut_start = int(original_n * start_percent)
        cut_end = int(original_n * end_percent)

        trimmed = patient_slices[cut_start: original_n - cut_end]

        #print(
        #    f"Patient {idx+1}: "
        #    f"Original={original_n} | After cut={len(trimmed)}"
        #)

        new_samples.extend(trimmed)

    # --------------------------------------
    # Step 3: Create new dataset object
    # --------------------------------------
    filtered_dataset = PairDataset.__new__(PairDataset)
    filtered_dataset.samples = new_samples

    return filtered_dataset    