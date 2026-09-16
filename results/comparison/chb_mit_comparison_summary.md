# CHB-MIT ARNN Comparison

## Preprocessed CSV evaluation

- Data: ictal_data.csv and preictal_data.csv
- Input channels: 23
- Test samples: 512
- Split: random segment-level 75/25 split
- Patients may appear in both training and testing
- Accuracy: 94.7266%
- F1: 0.946955
- PR-AUC: 0.983297
- ROC-AUC: 0.983936
- Confusion matrix: TN=244, FP=19, FN=8, TP=241

## Raw EDF evaluation

- Raw cases used: chb01 through chb12
- Training patients: chb01 through chb09
- Testing patients: chb10 through chb12
- Patient overlap: none
- Input channels: 18 standard bipolar EEG channels
- Segment length: 4 seconds
- Stride: 4 seconds
- Sampling rate: 256 Hz
- Training before balancing: 476,537 nonseizure and 943 seizure segments
- Training after balancing: 943 nonseizure and 943 seizure segments
- Test set: unchanged
- Test samples: 94950
- Accuracy: 93.8568%
- F1: 0.098036
- PR-AUC: 0.260323
- ROC-AUC: 0.795761
- Confusion matrix: TN=88800, FP=5558, FN=275, TP=317

## Difference: raw EDF minus preprocessed CSV

- Accuracy difference: -0.8698 percentage points
- F1 difference: -0.848919
- PR-AUC difference: -0.722975
- ROC-AUC difference: -0.188175

## Interpretation

The two results should not be treated as a direct one-to-one accuracy comparison. The preprocessed CSV experiment used a random segment-level split with 512 test samples. The raw EDF experiment used a stricter patient-wise split with 94,950 unchanged test segments. The raw test set is also highly imbalanced. Therefore, F1, PR-AUC, ROC-AUC, and the confusion matrix are more informative than accuracy alone for the raw EDF experiment.
