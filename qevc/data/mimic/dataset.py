"""MIMIC-III dataset loaders for the QEVC pipeline.

Two dataset classes:
  - MIMICRawDataset: loads raw MIMIC-III CSV tables for embedding extraction
  - MIMICDataset: loads pre-extracted PCA-fused features for training/evaluation

Task: In-hospital mortality prediction (binary classification).
  - Structured features: vital signs, lab values, demographics from CHARTEVENTS
  - Language features: discharge summaries from NOTEEVENTS
  - Groups: gender (M=0, F=1) for fairness evaluation

Designed for BU SCC where MIMIC-III CSVs are stored on project disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ── Paths for BU SCC ──
_SCC_PROJECT = Path("/projectnb/llm-plastsurg/kms_new")
_DEFAULT_MIMIC_DIR = _SCC_PROJECT / "project-imp" / "qevc" / "data" / "mimic"


# ── Key MIMIC-III tables ──
_REQUIRED_TABLES = [
    "ADMISSIONS.csv",
    "PATIENTS.csv",
    "CHARTEVENTS.csv",   # vital signs
    "LABEVENTS.csv",     # lab values
    "NOTEEVENTS.csv",    # clinical notes
]

# ── Vital sign ITEMID codes (from MIMIC-III documentation) ──
# These are the most commonly used vital signs for mortality prediction
VITAL_SIGN_ITEMIDS = {
    "heart_rate":       [211, 220045],
    "systolic_bp":      [51, 442, 455, 6701, 220179, 220050],
    "diastolic_bp":     [8368, 8440, 8441, 8555, 220180, 220051],
    "mean_bp":          [456, 52, 6702, 443, 220052, 220181, 225312],
    "respiratory_rate": [615, 618, 220210, 224690],
    "temperature":      [223761, 678, 223762, 676],
    "spo2":             [646, 220277],
    "glucose":          [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
}

# ── Lab value ITEMID codes ──
LAB_ITEMIDS = {
    "creatinine":       [50912],
    "bun":              [51006],
    "sodium":           [50983, 50824],
    "potassium":        [50971, 50822],
    "bicarbonate":      [50882],
    "chloride":         [50902, 50806],
    "hematocrit":       [51221, 50810],
    "hemoglobin":       [51222, 50811],
    "platelet":         [51265],
    "wbc":              [51301, 51300],
    "lactate":          [50813],
}

# Flatten all item IDs for filtering
ALL_VITAL_IDS = [iid for ids in VITAL_SIGN_ITEMIDS.values() for iid in ids]
ALL_LAB_IDS = [iid for ids in LAB_ITEMIDS.values() for iid in ids]


# ======================================================================== #
# MIMICRawDataset — for embedding extraction
# ======================================================================== #

class MIMICRawDataset:
    """Loads raw MIMIC-III tables and prepares data for embedding extraction.

    Handles:
      - Loading and merging ADMISSIONS + PATIENTS for demographics
      - Extracting vital signs from CHARTEVENTS
      - Extracting lab values from LABEVENTS
      - Loading discharge summaries from NOTEEVENTS
      - Computing in-hospital mortality labels

    Parameters
    ----------
    data_dir : Path
        Directory containing MIMIC-III CSV files (or symlinks to them).
    mimic_csv_dir : Path or None
        Override directory for raw MIMIC CSV files. If None, looks in
        ``data_dir/raw/`` or the PhysioNet default location.
    """

    def __init__(
        self,
        data_dir: Path,
        mimic_csv_dir: Optional[Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Find MIMIC CSV files
        if mimic_csv_dir is not None:
            self.csv_dir = Path(mimic_csv_dir)
        elif (self.data_dir / "raw").exists():
            self.csv_dir = self.data_dir / "raw"
        elif (_SCC_PROJECT / "mimic-iii" / "1.4").exists():
            self.csv_dir = _SCC_PROJECT / "mimic-iii" / "1.4"
        else:
            self.csv_dir = self.data_dir

        print(f"MIMIC-III CSV directory: {self.csv_dir}")

        # Load base tables
        self._admissions = None
        self._patients = None
        self._cohort = None

    @property
    def admissions(self) -> pd.DataFrame:
        """Lazy-load ADMISSIONS table."""
        if self._admissions is None:
            path = self.csv_dir / "ADMISSIONS.csv"
            if not path.exists():
                path = self.csv_dir / "ADMISSIONS.csv.gz"
            print(f"Loading ADMISSIONS from {path}...")
            self._admissions = pd.read_csv(
                path,
                parse_dates=["ADMITTIME", "DISCHTIME", "DEATHTIME"],
                usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME",
                          "DEATHTIME", "HOSPITAL_EXPIRE_FLAG", "ADMISSION_TYPE",
                          "ETHNICITY"],
            )
            print(f"  Loaded {len(self._admissions):,} admissions")
        return self._admissions

    @property
    def patients(self) -> pd.DataFrame:
        """Lazy-load PATIENTS table."""
        if self._patients is None:
            path = self.csv_dir / "PATIENTS.csv"
            if not path.exists():
                path = self.csv_dir / "PATIENTS.csv.gz"
            print(f"Loading PATIENTS from {path}...")
            self._patients = pd.read_csv(
                path,
                parse_dates=["DOB", "DOD"],
                usecols=["SUBJECT_ID", "GENDER", "DOB", "DOD"],
            )
            print(f"  Loaded {len(self._patients):,} patients")
        return self._patients

    @property
    def cohort(self) -> pd.DataFrame:
        """Build the study cohort: adult ICU admissions with mortality labels."""
        if self._cohort is None:
            # Merge admissions with patients
            df = self.admissions.merge(self.patients, on="SUBJECT_ID", how="left")

            # Compute approximate age at admission
            df["AGE"] = (
                (df["ADMITTIME"] - df["DOB"]).dt.total_seconds() / (365.25 * 24 * 3600)
            )

            # Filter to adults (age >= 18) and exclude neonates
            # MIMIC encodes patients >89 with shifted DOB → age > 300
            df = df[(df["AGE"] >= 18) & (df["AGE"] < 300)].copy()

            # Binary mortality label
            df["LABEL"] = df["HOSPITAL_EXPIRE_FLAG"].astype(int)

            # Gender group for fairness (M=0, F=1)
            df["GROUP"] = (df["GENDER"] == "F").astype(int)

            # Sort by admission time
            df = df.sort_values("ADMITTIME").reset_index(drop=True)

            self._cohort = df
            n_pos = df["LABEL"].sum()
            print(f"  Cohort: {len(df):,} admissions | "
                  f"Mortality: {n_pos:,} ({100 * n_pos / len(df):.1f}%)")
        return self._cohort

    def build_structured_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract structured features (vitals + labs + demographics).

        For each admission, computes aggregate statistics (mean, std, min, max,
        last) of vital signs and lab values during the first 48 hours.

        Returns
        -------
        features : ndarray (N, n_features)
        labels : ndarray (N,)
        groups : ndarray (N,)
        """
        cohort = self.cohort
        hadm_ids = cohort["HADM_ID"].values

        # ── Load CHARTEVENTS (vital signs) ──
        print("Loading CHARTEVENTS (vital signs)...")
        chart_path = self.csv_dir / "CHARTEVENTS.csv"
        if not chart_path.exists():
            chart_path = self.csv_dir / "CHARTEVENTS.csv.gz"

        # Read in chunks to handle the massive file
        chart_chunks = []
        hadm_set = set(hadm_ids)
        for chunk in pd.read_csv(
            chart_path,
            chunksize=1_000_000,
            usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
            dtype={"HADM_ID": "Int64", "ITEMID": int},
        ):
            # Filter to our cohort and relevant item IDs
            mask = (
                chunk["HADM_ID"].isin(hadm_set) &
                chunk["ITEMID"].isin(ALL_VITAL_IDS) &
                chunk["VALUENUM"].notna()
            )
            if mask.any():
                chart_chunks.append(chunk[mask])

        if chart_chunks:
            chart_df = pd.concat(chart_chunks, ignore_index=True)
            print(f"  Filtered to {len(chart_df):,} vital sign measurements")
        else:
            chart_df = pd.DataFrame(columns=["HADM_ID", "ITEMID", "VALUENUM"])
            print("  WARNING: No vital sign measurements found")

        # ── Load LABEVENTS ──
        print("Loading LABEVENTS (lab values)...")
        lab_path = self.csv_dir / "LABEVENTS.csv"
        if not lab_path.exists():
            lab_path = self.csv_dir / "LABEVENTS.csv.gz"

        lab_chunks = []
        for chunk in pd.read_csv(
            lab_path,
            chunksize=1_000_000,
            usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
            dtype={"HADM_ID": "Int64", "ITEMID": int},
        ):
            mask = (
                chunk["HADM_ID"].isin(hadm_set) &
                chunk["ITEMID"].isin(ALL_LAB_IDS) &
                chunk["VALUENUM"].notna()
            )
            if mask.any():
                lab_chunks.append(chunk[mask])

        if lab_chunks:
            lab_df = pd.concat(lab_chunks, ignore_index=True)
            print(f"  Filtered to {len(lab_df):,} lab measurements")
        else:
            lab_df = pd.DataFrame(columns=["HADM_ID", "ITEMID", "VALUENUM"])
            print("  WARNING: No lab measurements found")

        # ── Aggregate features per admission ──
        print("Aggregating features per admission...")
        all_events = pd.concat([chart_df, lab_df], ignore_index=True)

        # Map ITEMID to feature name
        itemid_to_name = {}
        for name, ids in {**VITAL_SIGN_ITEMIDS, **LAB_ITEMIDS}.items():
            for iid in ids:
                itemid_to_name[iid] = name

        all_events["FEATURE_NAME"] = all_events["ITEMID"].map(itemid_to_name)

        # Compute aggregates: mean, std, min, max, last per feature per admission
        agg_funcs = ["mean", "std", "min", "max", "last"]
        pivot = (
            all_events
            .groupby(["HADM_ID", "FEATURE_NAME"])["VALUENUM"]
            .agg(agg_funcs)
            .unstack(level="FEATURE_NAME")
        )

        # Flatten multi-level columns
        pivot.columns = [f"{feat}_{stat}" for stat, feat in pivot.columns]

        # Reindex to match cohort
        features_df = pivot.reindex(hadm_ids).fillna(0.0)

        # Add demographics
        features_df["age"] = cohort["AGE"].values
        features_df["is_female"] = cohort["GROUP"].values

        # Convert to numpy
        features = features_df.values.astype(np.float32)
        labels = cohort["LABEL"].values.astype(np.int64)
        groups = cohort["GROUP"].values.astype(np.int64)

        # Normalize features (z-score)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        print(f"  Final features shape: {features.shape}")
        return features, labels, groups

    def load_notes(self) -> pd.DataFrame:
        """Load NOTEEVENTS table (discharge summaries).

        Returns
        -------
        DataFrame with columns: SUBJECT_ID, HADM_ID, CATEGORY, TEXT
        """
        path = self.csv_dir / "NOTEEVENTS.csv"
        if not path.exists():
            path = self.csv_dir / "NOTEEVENTS.csv.gz"

        print(f"Loading NOTEEVENTS from {path}...")
        notes = pd.read_csv(
            path,
            usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
        )
        print(f"  Loaded {len(notes):,} notes")
        return notes

    def get_discharge_summaries(self) -> dict[int, str]:
        """Get one discharge summary per HADM_ID.

        Returns
        -------
        dict mapping HADM_ID → discharge summary text (truncated to 512 chars)
        """
        notes = self.load_notes()
        discharge = notes[notes["CATEGORY"] == "Discharge summary"]
        summaries = (
            discharge
            .sort_values("HADM_ID")
            .groupby("HADM_ID")["TEXT"]
            .first()
            .to_dict()
        )
        # Truncate for RoBERTa
        return {k: v[:512] if isinstance(v, str) else "" for k, v in summaries.items()}


# ======================================================================== #
# MIMICDataset — for training on pre-extracted PCA features
# ======================================================================== #

class MIMICDataset(Dataset):
    """PyTorch Dataset that loads pre-extracted PCA-fused features for MIMIC.

    Used by ``train_qevc.py`` and ``run_baselines.py`` after embeddings
    have been extracted and PCA-fused.

    Expects files in data_dir:
        - fused_{split}.npy     — (N, n_pca) float32 features
        - meta_{split}.npz      — labels (N,) int64, groups (N,) int64

    Parameters
    ----------
    data_dir : Path
        Directory with pre-extracted data.
    split : str
        ``'train'`` or ``'test'``.
    n_samples : int or None
        Limit dataset size (for sanity checks).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        n_samples: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        # Load features
        fused_path = self.data_dir / f"fused_{split}.npy"
        meta_path = self.data_dir / f"meta_{split}.npz"

        self.features = np.load(fused_path).astype(np.float32)
        meta = np.load(meta_path)
        self.labels = meta["labels"].astype(np.int64)
        self.groups = meta["groups"].astype(np.int64)

        # Optional subsetting
        if n_samples is not None and n_samples < len(self.features):
            self.features = self.features[:n_samples]
            self.labels = self.labels[:n_samples]
            self.groups = self.groups[:n_samples]

        print(f"MIMICDataset({split}): {len(self)} samples, "
              f"{self.n_classes} classes, features={self.features.shape}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "group": torch.tensor(self.groups[idx], dtype=torch.long),
        }

    @property
    def n_classes(self) -> int:
        """Binary classification: survived (0) vs died (1)."""
        return 2
