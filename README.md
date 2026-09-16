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
