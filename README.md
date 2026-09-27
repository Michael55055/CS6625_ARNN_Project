# Original ARNN on Raw CHB-MIT EDF Data

This project applies the original ARNN architecture to raw CHB-MIT EDF recordings and compares the result with the two preprocessed CHB-MIT CSV files. No new model feature was added.

## Dataset scope

- The full CHB-MIT dataset has 24 cases.
- This project uses 12 cases: `chb01` through `chb12`.
- Valid EDF files: 354.
- Excluded files: `chb12_27.edf`, `chb12_28.edf`, and `chb12_29.edf` because they did not contain the selected 18-channel montage.
- Dataset source: https://physionet.org/content/chbmit/1.0.0/

Raw EEG data is not included in this repository.

## Raw-data processing

- Sampling rate: 256 Hz
- Segment length: 4 seconds
- Segment samples: 1,024
- Stride: 4 seconds
- Segment overlap: none
- Channels used: 18 standard bipolar EEG channels
- A segment is seizure-positive when it overlaps a seizure interval reported in the case summary file.
- For duplicate T8-P8 channels, the first copy, T8-P8-0, is used.

## Patient-wise split

- Training: `chb01` through `chb09`
- Testing: `chb10`, `chb11`, and `chb12`
- Patient overlap: none
- Random seed: 1111

This is different from the preprocessed CSV experiment, which uses a random segment-level 75/25 split.

## Class counts

| Split | Nonseizure | Seizure | Total |
|---|---:|---:|---:|
| Training before balancing | 476,537 | 943 | 477,480 |
| Training after balancing | 943 | 943 | 1,886 |
| Testing unchanged | 94,358 | 592 | 94,950 |

Only the training set is balanced. The test set is never undersampled or oversampled. Normalization values are calculated from the balanced training data only.

## ARNN settings

- Original ARNN architecture
- Epochs: 30
- Batch size: 50
- Initial learning rate: 0.001
- Optimizer: Adam
- Decision threshold: 0.5
- Input shape: `(1024, 18)`
- Embedding dimension: 40
- Attention heads: 4
- Recurrence steps: 16
- State vectors: 64

## Environment

- Google Colab
- Python 3.13
- PyTorch 2.11
- MNE 1.13
- NVIDIA Tesla T4 GPU

Install packages:

```bash
pip install -r requirements.txt
```

## 1. Download raw EDF data

```bash
python download_chbmit.py \
  --output_folder /path/to/raw_edf \
  --first_case 1 \
  --last_case 12
```

The download is large and may take many hours. The download script uses `wget -c`, so an interrupted file can resume.

## 2. Prepare model-ready segments

```bash
python prepare_raw_chbmit.py \
  --raw_folder /path/to/raw_edf \
  --output_folder /path/to/prepared_data \
  --seed 1111
```

## 3. Train ARNN

```bash
python train_raw_arnn.py \
  --cache_folder /path/to/prepared_data \
  --output_folder /path/to/training_output \
  --epochs 30 \
  --batch_size 50 \
  --learning_rate 0.001 \
  --seed 1111
```

## 4. Evaluate ARNN

```bash
python evaluate_raw_arnn.py \
  --cache_folder /path/to/prepared_data \
  --raw_edf_folder /path/to/raw_edf \
  --checkpoint /path/to/training_output/raw_arnn_epoch_30.pt \
  --output_folder /path/to/evaluation_output \
  --batch_size 50 \
  --threshold 0.5
```

## Results

| Metric | Preprocessed CSV | Raw EDF patient-wise |
|---|---:|---:|
| Test samples | 512 | 94,950 |
| Accuracy | 94.73% | 93.86% |
| F1 | 0.9470 | 0.0980 |
| PR-AUC | 0.9833 | 0.2603 |
| ROC-AUC | 0.9839 | 0.7958 |
| True negative | 244 | 88,800 |
| False positive | 19 | 5,558 |
| False negative | 8 | 275 |
| True positive | 241 | 317 |

The raw test set contains only 0.62% seizure segments. Therefore, accuracy alone is misleading. F1, PR-AUC, ROC-AUC, and the confusion matrix should be considered together.

The results are not a direct one-to-one comparison. The CSV experiment uses a small random segment split, while the raw EDF experiment tests on completely separate patients.

## Runtime

- Training time: approximately 177 seconds
- Evaluation time: approximately 335 seconds
- Hardware: NVIDIA Tesla T4

These times exclude downloading and preprocessing.

## Repository structure

```text
CHB_MIT/model.py
models/ARNN.py
download_chbmit.py
prepare_raw_chbmit.py
train_raw_arnn.py
evaluate_raw_arnn.py
requirements.txt
results/
README.md
```

## Excluded from GitHub

- Raw EDF recordings
- NumPy signal caches
- Trained model checkpoints
- Full test prediction files
- Credentials

## Original ARNN source

The original model architecture comes from:

https://github.com/Salim-Lysiun/ARNN

The original `model.py` and `ARNN.py` files are preserved. The new scripts provide the raw-data preparation, training, evaluation, and patient-wise comparison workflow.

---

## Raw EDF preictal-versus-ictal extension

This extension more closely matches the two classes in the original preprocessed CHB-MIT CSV files. It uses the unchanged original ARNN architecture.

### Class definitions

- Negative class: preictal
- Positive class: ictal
- Preictal interval: immediately before each seizure, with duration equal to that seizure
- Segment length: 4 seconds
- Stride: 4 seconds
- Segment overlap: none
- Sampling rate: 256 Hz
- Channels: 18 common bipolar EEG channels

### Patient-wise split

- Training: `chb01` through `chb09`
- Testing: `chb10`, `chb11`, and `chb12`
- Patient overlap: none
- Training preictal segments: 891
- Training ictal segments: 891
- Testing preictal segments: 524
- Testing ictal segments: 524
- Normalization statistics were calculated from training patients only
- Random seed: 1111
- Decision threshold: 0.5
- The test patients were not used to select the threshold

### Prepare model-ready segments

The audited manifest is included at:

```text
manifests/preictal_ictal/preictal_ictal_manifest.csv
```

```bash
python prepare_preictal_ictal_chbmit.py \
  --raw_folder /path/to/raw_edf \
  --manifest manifests/preictal_ictal/preictal_ictal_manifest.csv \
  --output_folder /path/to/prepared_data
```

### Train the original ARNN

```bash
python train_preictal_ictal_arnn.py \
  --cache_folder /path/to/prepared_data \
  --output_folder /path/to/training_output \
  --epochs 30 \
  --batch_size 50 \
  --learning_rate 0.001 \
  --seed 1111
```

### Evaluate on separate patients

```bash
python evaluate_preictal_ictal_arnn.py \
  --cache_folder /path/to/prepared_data \
  --checkpoint /path/to/training_output/raw_arnn_epoch_30.pt \
  --output_folder /path/to/evaluation_output \
  --batch_size 50 \
  --threshold 0.5
```

The evaluator works with either CPU or GPU.

### Patient-wise test results

| Metric | Result |
|---|---:|
| Test samples | 1,048 |
| Accuracy | 75.57% |
| F1 | 0.7241 |
| PR-AUC | 0.8628 |
| ROC-AUC | 0.8342 |
| Sensitivity | 0.6412 |
| Specificity | 0.8702 |
| True negative | 456 |
| False positive | 68 |
| False negative | 188 |
| True positive | 336 |

### Interpretation

The preprocessed CSV experiment achieved higher performance, but it used a random segment-level split.

The raw EDF preictal-versus-ictal experiment used completely separate patients for testing. Its lower performance shows that generalization to unseen patients is more difficult.

The raw nonseizure-versus-ictal experiment is a different seizure-detection task. It is not a direct reproduction of the CSV experiment because the negative-class definitions are different.

Detailed files are available under:

```text
results/preictal_ictal/
```


## Final Assignment Analysis

The final milestone evaluates the original ARNN architecture under three separate CHB-MIT protocols and includes the completed UPenn/Mayo reproduction as a separate baseline study.

### CHB-MIT protocols

| Experiment | Classes | Evaluation split | Test composition |
|---|---|---|---|
| Preprocessed CSV baseline | Preictal vs. ictal | Original random 75/25 segment-level split | 263 preictal and 249 ictal segments |
| Raw-EDF detection | All nonseizure vs. ictal | chb01-chb09 training; chb10-chb12 testing | 94,358 nonseizure and 592 ictal segments |
| Raw-EDF matched classes | Preictal vs. ictal | chb01-chb09 training; chb10-chb12 testing | 524 preictal and 524 ictal segments |

The experiments must be interpreted separately. They differ in split policy, preprocessing, segment selection, class definitions, and test prevalence. Therefore, score differences cannot be attributed to the patient split alone.

### Preictal definition

For the raw-EDF matched-classes experiment, preictal is defined as the interval immediately before each seizure. Its retained duration is matched to that seizure's duration using complete non-overlapping four-second segments. This is an event-matched operational definition and not a fixed advance-warning horizon.

A seizure event is excluded when a complete valid preictal interval cannot be created, when the interval overlaps another seizure, or when the seizure EDF lacks the selected 18-channel montage. The audit contains 98 seizure events: 84 included and 14 excluded. Thirteen exclusions resulted from EDF files that failed the 18-channel audit, and one resulted from a preictal interval overlapping another seizure.

### Main CHB-MIT results

| Experiment | Accuracy | Ictal precision | Ictal sensitivity | F1 | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Preprocessed CSV baseline | 0.9473 | 0.9269 | 0.9679 | 0.9470 | 0.9833 | 0.9839 |
| Raw-EDF detection | 0.9386 | 0.0540 | 0.5355 | 0.0980 | 0.2603 | 0.7958 |
| Raw-EDF matched classes | 0.7557 | 0.8317 | 0.6412 | 0.7241 | 0.8628 | 0.8342 |

The raw-EDF detection test set is naturally imbalanced. An always-nonseizure classifier would obtain 99.38% accuracy, which is higher than the model's 93.86% accuracy. Therefore, accuracy alone is misleading for this experiment.

### Patient and event analysis

In the matched-classes experiment, ictal sensitivity was 0.8165 for chb10, 0.9353 for chb11, and 0.2757 for chb12. The lower sensitivity for chb12 shows substantial variation across unseen patients.

The test set contains 36 included seizure events. Median event-level ictal sensitivity was 0.3417. The long seizure in chb11_99.edf contributed 188 ictal and 188 matched preictal segments, or 376 of the 1,048 test segments. Its ictal sensitivity was 0.9362. When this event was omitted in a post hoc supplementary check, pooled ictal sensitivity decreased from 0.6412 to 0.4762. The full 1,048-segment result remains the primary result, and neither test analysis was used for threshold selection.

### Decision threshold

All required CHB-MIT results use the prespecified decision threshold of 0.5. No threshold was selected using chb10-chb12.

### UPenn/Mayo baseline

The UPenn/Mayo reproduction contains 120 completed runs: 10 runs for each of Dog_1-Dog_4 and Patient_1-Patient_8. It uses the original participant-specific random segment-level split and is not a held-out-patient evaluation. Only final accuracy was verifiable from the saved run outputs. Unavailable metrics are reported as NA in results_week9.csv.

### Final-assignment outputs

- `results/final_assignment/results_week9.csv`: three CHB-MIT experiment rows and 120 UPenn/Mayo participant-run rows.
- `results/final_assignment/chb_mit_event_analysis.csv`: one row for each included test seizure.
- `results/final_assignment/chb_mit_event_analysis_summary.csv`: pooled, median-event, and long-event sensitivity results.
- `results/final_assignment/chb_mit_exclusion_summary.csv`: excluded-event counts and reasons.
- `results/final_assignment/preictal_ictal_metrics_by_patient.csv`: chb10, chb11, and chb12 results.
- `results/final_assignment/preictal_ictal_pr_curve.png`: labeled precision-recall curve.
- `results/final_assignment/preictal_ictal_confusion_matrix.png`: labeled confusion matrix.

### Reproduce the event analysis

```bash
python analyze_chb_mit_events.py \
  --predictions results/preictal_ictal/preictal_ictal_test_predictions.csv \
  --event_audit manifests/preictal_ictal/preictal_ictal_event_audit.csv \
  --output_folder results/final_assignment \
  --threshold 0.5 \
  --long_event_id chb11_event_003
```

Raw EDF recordings, NumPy signal caches, trained checkpoints, and credentials are intentionally omitted because of dataset restrictions, size, and security considerations.
