"""
EEG analysis pipeline for BioSemi BDF files using MNE-Python.
The script provides modular functions to load data, preprocess, epoch,
compute ERP and N2pc, and optionally run time-frequency analysis and
multivariate decoding.

Requirements:
    - Python 3.9+
    - mne
    - numpy, scipy, matplotlib
    - scikit-learn (for decoding)

This script is a template and may require adaptation to specific
recording setups.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import mne
import numpy as np

try:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from mne.decoding import SlidingEstimator, Vectorizer
except Exception:
    # scikit-learn may not be installed in minimal environments
    SlidingEstimator = None


def log(msg: str):
    """Simple logger."""
    print(f"[eeg_analysis] {msg}")


@dataclass
class Config:
    bdf_path: Path
    subject: str
    event_id: Dict[str, int]
    tmin: float = -0.2
    tmax: float = 0.8
    l_freq: float = 0.1
    h_freq: float = 40.0
    montage: str = "standard_1020"
    n_jobs: int = 1
    derivatives_dir: Path = Path("derivatives")
    ica_method: str = "fastica"  # or "picard"
    do_tfr: bool = False
    do_decoding: bool = False
    baseline: Optional[tuple] = (None, 0)

    def subject_dir(self) -> Path:
        return self.derivatives_dir / self.subject


# ---------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------

def load_data(cfg: Config) -> mne.io.BaseRaw:
    """Load BDF file and set channel types."""
    log("Loading raw data")
    raw = mne.io.read_raw_bdf(cfg.bdf_path, preload=True)
    raw.set_montage(cfg.montage)

    # Try to set EOG/ECG channel types if present
    for ch in raw.ch_names:
        if ch.upper().startswith("EOG"):
            raw.set_channel_types({ch: "eog"})
        if ch.upper().startswith("ECG") or "EKG" in ch.upper():
            raw.set_channel_types({ch: "ecg"})
    return raw


def preprocess(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Basic preprocessing: filtering, referencing, bad channels."""
    log("Running preprocessing")
    raw.filter(cfg.l_freq, cfg.h_freq, n_jobs=cfg.n_jobs)

    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        log("Referencing to mastoids")
        raw.set_eeg_reference(["M1", "M2"])
    else:
        log("Using average reference")
        raw.set_eeg_reference("average", projection=False)

    # Detect bad channels using a PSD z-score approach
    psd = raw.compute_psd(fmin=1, fmax=40)
    psd_mean = psd.get_data().mean(axis=-1)
    z = (psd_mean - np.mean(psd_mean)) / np.std(psd_mean)
    bads = [raw.ch_names[i] for i, val in enumerate(z) if np.abs(val) > 3.0]
    if bads:
        log(f"Detected bad channels: {bads}")
        raw.info["bads"] = list(set(raw.info.get("bads", []) + bads))
        raw.interpolate_bads(reset_bads=True)
    return raw


# ---------------------------------------------------------------------
# ICA for artifact correction
# ---------------------------------------------------------------------

def run_ica(raw: mne.io.BaseRaw, cfg: Config) -> mne.preprocessing.ICA:
    log("Fitting ICA")
    ica = mne.preprocessing.ICA(method=cfg.ica_method, random_state=42)
    ica.fit(raw)

    log("Detecting artifact components")
    eog_inds, _ = ica.find_bads_eog(raw)
    ecg_inds, _ = ica.find_bads_ecg(raw)
    ica.exclude = list(set(eog_inds + ecg_inds))
    log(f"Excluding components: {ica.exclude}")

    ica.apply(raw)
    return ica


# ---------------------------------------------------------------------
# Epoching and ERP
# ---------------------------------------------------------------------

def epoch_data(raw: mne.io.BaseRaw, cfg: Config) -> mne.Epochs:
    log("Epoching data")
    if raw.annotations:
        events, event_id = mne.events_from_annotations(raw)
    else:
        events = mne.find_events(raw)
        event_id = cfg.event_id

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=cfg.baseline,
        preload=True,
        reject_by_annotation=True,
    )

    epochs.drop_bad("peak_to_peak")
    return epochs


# ---------------------------------------------------------------------
# N2pc computation
# ---------------------------------------------------------------------

def compute_n2pc(
    epochs: mne.Epochs,
    left_event: str,
    right_event: str,
    cfg: Config,
) -> mne.Evoked:
    log("Computing N2pc")
    picks = [ch for ch in ("PO7", "PO8") if ch in epochs.ch_names]
    if len(picks) < 2:
        raise RuntimeError("PO7/PO8 channels not found")

    left = epochs[left_event].average()
    right = epochs[right_event].average()

    po7 = epochs.ch_names.index("PO7")
    po8 = epochs.ch_names.index("PO8")

    diff_left = left.data[po8] - left.data[po7]
    diff_right = right.data[po7] - right.data[po8]
    n2pc_data = (diff_left + diff_right) / 2

    evoked = mne.EvokedArray(n2pc_data[np.newaxis, :], left.info, tmin=left.times[0])
    evoked.comment = "N2pc"
    return evoked


# ---------------------------------------------------------------------
# Time-frequency analysis (optional)
# ---------------------------------------------------------------------

def compute_tfr(epochs: mne.Epochs, cfg: Config) -> mne.time_frequency.EpochsTFR:
    log("Computing time-frequency representation")
    freqs = np.logspace(np.log10(4), np.log10(30), 20)
    n_cycles = freqs / 2.0
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=3,
        n_jobs=cfg.n_jobs,
    )
    return tfr


# ---------------------------------------------------------------------
# Multivariate decoding (optional)
# ---------------------------------------------------------------------

def run_decoding(epochs: mne.Epochs, cfg: Config):
    if SlidingEstimator is None:
        log("scikit-learn not installed; skipping decoding")
        return None

    log("Running decoding")
    X = epochs.get_data()
    y = epochs.events[:, 2]

    clf = make_pipeline(
        Vectorizer(), StandardScaler(), LinearDiscriminantAnalysis()
    )
    time_decod = SlidingEstimator(clf, n_jobs=cfg.n_jobs, scoring="roc_auc")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(time_decod, X, y, cv=cv)
    log(f"Decoding mean AUC: {np.mean(scores):.3f}")
    return scores


# ---------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main(cfg: Config):
    subj_dir = cfg.subject_dir()
    ensure_dir(subj_dir)

    raw = load_data(cfg)
    raw = preprocess(raw, cfg)
    ica = run_ica(raw, cfg)

    epochs = epoch_data(raw, cfg)

    # Save processed objects
    raw_path = subj_dir / f"{cfg.subject}_raw.fif"
    ica_path = subj_dir / f"{cfg.subject}_ica.fif"
    epochs_path = subj_dir / f"{cfg.subject}_epochs.fif"

    log(f"Saving raw to {raw_path}")
    raw.save(raw_path, overwrite=True)
    log(f"Saving ICA to {ica_path}")
    ica.save(ica_path)
    log(f"Saving epochs to {epochs_path}")
    epochs.save(epochs_path, overwrite=True)

    evokeds = {cond: epochs[cond].average() for cond in cfg.event_id}
    for cond, evo in evokeds.items():
        evo_path = subj_dir / f"{cfg.subject}_{cond}-ave.fif"
        log(f"Saving evoked {cond} to {evo_path}")
        evo.save(evo_path)

    # N2pc computation
    if set(["left", "right"]).issubset(cfg.event_id):
        n2pc = compute_n2pc(epochs, "left", "right", cfg)
        n2pc_path = subj_dir / f"{cfg.subject}_n2pc-ave.fif"
        log(f"Saving N2pc to {n2pc_path}")
        n2pc.save(n2pc_path)

        fig = n2pc.plot(spatial_colors=True, show=False)
        fig_path = subj_dir / f"{cfg.subject}_n2pc.png"
        fig.savefig(fig_path)
        log(f"Figure saved to {fig_path}")

    # Optional time-frequency analysis
    if cfg.do_tfr:
        tfr = compute_tfr(epochs, cfg)
        tfr_path = subj_dir / f"{cfg.subject}_tfr.h5"
        log(f"Saving TFR to {tfr_path}")
        tfr.save(tfr_path, overwrite=True)
        tfr_fig = tfr.plot_average(show=False)
        tfr_fig[0].savefig(subj_dir / f"{cfg.subject}_tfr.png")

    # Optional decoding
    if cfg.do_decoding:
        scores = run_decoding(epochs, cfg)
        if scores is not None:
            dec_path = subj_dir / "decoding_results.txt"
            dec_path.write_text("\n".join(map(str, scores)))


if __name__ == "__main__":
    example_cfg = Config(
        bdf_path=Path("S01_VsGaze.bdf"),
        subject="S01",
        event_id={"left": 1, "right": 2},
        do_tfr=False,
        do_decoding=False,
    )
    main(example_cfg)
