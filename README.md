# Test-Codex

This repository contains a minimal EEG analysis example.

## EEG analysis pipeline

The `eeg_analysis.py` script implements a modular pipeline for processing
BioSemi `.BDF` files with MNE-Python. It includes functions for loading
raw data, preprocessing, ICA, epoching, N2pc computation and optional
time–frequency analysis and decoding. Adjust the `Config` object at the
bottom of the script to match your dataset before running:

```bash
python eeg_analysis.py
```
