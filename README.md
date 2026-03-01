# ECVAD-Lite: Event-Conditioned Video Anomaly Detection (VAD)

A compact codebase for **event-guided video anomaly detection** in challenging conditions (high-speed motion, low frame-rate RGB capture).

## Core idea
This project is organized around two innovations:

1. **Event-conditioned corresponding-frame reconstruction** as a VAD front-end correction module.
   - Use event evidence to reconstruct clearer motion-aware RGB observations at target timestamps.
   - Reduce false positives from motion blur / temporal under-sampling.

2. **Event-consistency constrained credible anomaly scoring**.
   - Use event support as reliability evidence to reweight reconstruction error.
   - Emphasize error where events support true motion/brightness change.
   - Penalize hallucinated changes in event-free areas.

## Repository scope (slim)
This branch intentionally keeps only the core Python parts:

- `dataset/`: frame-aligned RGB/event loading.
- `models/`: minimal training/validation wrappers and VAD scoring path.
- `losses/`: reconstruction losses + event-consistency VAD scoring loss.
- `run_network.py`: lightweight entrypoint.

## Expected data format
Current loader assumes:

- RGB and event files are **frame-aligned** (`len(rgb)==len(events)` per sequence).
- Event files are `.npz` with key `data` (shape `[C,H,W]` or `[H,W]`).

## Notes
This is a research skeleton focused on method logic, not a full production training harness.
