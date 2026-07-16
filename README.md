# EMG-Based Hand Gesture Classification

A signal processing and machine learning pipeline that classifies hand gestures from surface EMG (sEMG) muscle signals, using the public Ninapro DB1 dataset. This is the same underlying technology behind myoelectric prosthetic hand control.

## Motivation

Muscle contractions produce small electrical signals that can be picked up by surface electrodes. If a machine can learn to recognize which gesture a person is attempting purely from that electrical signal, it can control a device — most notably, a prosthetic hand — without any other input. This project builds and evaluates that classification pipeline from raw signal to predicted gesture, using an approach grounded in evidence gathered from the data itself rather than assumptions from a textbook.

## Dataset

**Ninapro DB1** — 27 subjects, 52 hand/wrist movements, recorded with 10 Otto Bock electrodes at 100 Hz. This project uses a subset of subjects and exercise sessions for development and evaluation.

## Pipeline

1. **Load** — raw EMG (10 channels) and gesture labels (`restimulus`) from `.mat` files.
2. **Window** — the continuous signal is split into 200ms windows (20 samples) with 50% overlap (10-sample step).
3. **Label** — each window is labeled using majority vote across its samples (not a single midpoint sample — see below).
4. **Feature extraction** — per channel, per window: Mean, Standard Deviation, and Max (30 features per window across 10 channels).
5. **Class balancing** — during training only, "rest" windows are downsampled (kept 1-in-10) to prevent the classifier from defaulting to the majority class.
6. **Scaling** — features are standardized (mean 0, std 1) using parameters learned only from the training set, then applied to test data — never fit on test data.
7. **Classification** — Support Vector Machine with an RBF kernel.
8. **Evaluation** — accuracy on held-out data, tested both within-subject (different session, same person) and cross-subject.

## Key Engineering Decisions

**Why there's no bandpass filter.** The standard EMG filtering range (20–450 Hz) doesn't apply to this dataset. Ninapro DB1's electrodes (Otto Bock Myobock) perform hardware-level rectification and smoothing before the signal is ever recorded — verified directly by inspecting raw sample values, which are entirely non-negative and smoothly varying, unlike true raw EMG's rapid oscillation around zero. Applying a textbook bandpass filter to this already-processed envelope signal measurably reduced performance in testing. Filtering was removed after this was confirmed with evidence, not assumed.

**Why `restimulus`, not `stimulus`.** The dataset's `stimulus` field marks when a gesture cue appeared on screen — but human reaction time (roughly 300ms–1000ms) means the muscle hasn't actually started moving yet at that exact moment. `restimulus` is Ninapro's corrected label, adjusted for this delay, giving more accurate ground truth.

**Why majority-vote labeling.** Labeling an entire 200ms window using only the sample at its midpoint risks mislabeling windows that straddle a transition between rest and a gesture. Taking the majority label across all samples in the window is more robust to this boundary noise.

**Why per-subject scaling.** Fitting a single scaler on one subject and applying it to a different subject was tested and found to significantly *hurt* cross-subject accuracy — different people have different baseline EMG amplitude due to skin impedance, electrode placement, and muscle physiology. Scaling parameters are only ever learned from the set they're applied to (training set for training/same-subject test; a subject's own data for cross-subject comparisons).

**Why rest-class downsampling only applies to training data.** Roughly half of any recording is "rest." Without addressing this, a classifier can achieve deceptively high accuracy simply by guessing "rest" often. Downsampling only the training set's rest windows forces the model to learn to distinguish the 12 actual gestures. The test set is always left fully intact and untouched, to keep evaluation honest and realistic.

## Results

| Configuration | Same-subject accuracy | Cross-subject accuracy |
|---|---|---|
| Baseline: bandpass filter, MAV/RMS/ZC features, Logistic Regression | 39.7% | 30.8% |
| Filter removed | 40.4% | 17.0% |
| Filter removed + Mean/Std/Max features | 40.6% | 14.5% |
| **`restimulus` + majority-vote labeling + per-subject scaling + SVM (RBF)** | **61.7%** | *not yet tested* |

(Random chance baseline across 13 classes ≈ 7.7%.)

**Tested and found not to help:** Random Forest (overfit given limited training data — cross-subject accuracy dropped to 11.5%); widening the filter range toward the true Nyquist limit (cross-subject accuracy dropped further, same-subject unchanged); scaling with a single scaler fit across both subjects (cross-subject accuracy dropped to 5.8%).

## Limitations

- **Cross-subject generalization remains a significant challenge**, consistent with published literature on this exact problem — EMG signal characteristics vary substantially between individuals due to physiology and electrode placement. Naive approaches trained on a single subject are known to generalize poorly; techniques like domain adaptation (e.g., AdaBN) exist specifically to address this and are a natural next step.
- Trained on a limited number of subjects; more training subjects would likely improve generalization.
- Feature set (Mean, Std, Max) is intentionally simple; richer time-domain features (Waveform Length, Slope Sign Changes) are a documented next step.

## Future Work

- Extend training to more subjects to improve cross-subject generalization.
- Simulate electrode displacement/noise robustness (a known real-world wearable failure mode).
- Explore domain adaptation techniques for cross-subject transfer.

## Tech Stack

Python, NumPy, SciPy (signal processing, `.mat` file I/O), scikit-learn (SVM, preprocessing, evaluation).

## Author

Moosa Hamza — Biomedical Engineering, UBC
