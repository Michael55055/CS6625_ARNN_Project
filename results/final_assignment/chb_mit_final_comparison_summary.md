# Final CHB-MIT ARNN Comparison

## Experimental protocols

### Preprocessed CSV experiment

- Classification: preictal versus ictal
- Split: random 75/25 segment-level split
- Test samples: 512
- This split is not patient-independent.

### Raw EDF seizure-detection experiment

- Classification: all nonseizure segments versus ictal segments
- Training patients: chb01 through chb09
- Testing patients: chb10 through chb12
- Test samples: 94950
- Seizure segments: 592
- All-nonseizure baseline accuracy: 0.9938
- This experiment is not directly comparable with the CSV experiment because the negative classes are different.

### Raw EDF preictal-versus-ictal experiment

- Classification: preictal versus ictal
- Preictal interval: immediately before each seizure, with duration equal to the seizure
- Training patients: chb01 through chb09
- Testing patients: chb10 through chb12
- Patient overlap: none
- Test samples: 1048
- Preictal segments: 524
- Ictal segments: 524
- Threshold: 0.5
- The threshold was not selected using test patients.

## Core results

| Experiment | Accuracy | F1 | PR-AUC | ROC-AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Preprocessed CSV | 0.9473 | 0.9470 | 0.9833 | 0.9839 | 0.9679 | 0.9278 |
| Raw nonseizure vs ictal | 0.9386 | 0.0980 | 0.2603 | 0.7958 | 0.5355 | 0.9411 |
| Raw preictal vs ictal | 0.7557 | 0.7241 | 0.8628 | 0.8341 | 0.6412 | 0.8702 |

## Interpretation

The raw preictal-versus-ictal experiment more closely matches the class definitions used by the original CSV experiment. However, it is still not a direct one-to-one reproduction because it uses a patient-wise split, raw EDF processing, 18 common EEG channels, and completely separate test patients.

The lower patient-wise test performance is an important result. It indicates that the original ARNN has more difficulty generalizing to unseen patients than the random CSV split suggests.

The raw nonseizure-versus-ictal result addresses a different seizure-detection problem. Its accuracy should not be emphasized because the test set is naturally imbalanced and an all-nonseizure classifier would achieve higher accuracy.
