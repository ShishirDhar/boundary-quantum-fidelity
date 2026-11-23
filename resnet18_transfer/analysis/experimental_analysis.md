# Experimental Analysis: ResNet18 Baselines on HAM10000

## Test-set performance summary


### Ordinary CE
- Accuracy: 0.8165
- Balanced accuracy: 0.5632
- Macro F1: 0.5985

Per-class metrics:
  - Melanocytic nevus (nv): F1=0.921, Prec=0.884, Rec=0.963, PR-AUC=0.976
  - Melanoma (mel): F1=0.348, Prec=0.588, Rec=0.247, PR-AUC=0.437
  - Benign keratosis (bkl): F1=0.625, Prec=0.699, Rec=0.565, PR-AUC=0.670
  - Basal cell carcinoma (bcc): F1=0.628, Prec=0.559, Rec=0.717, PR-AUC=0.770
  - Actinic keratosis (akiec): F1=0.543, Prec=0.512, Rec=0.579, PR-AUC=0.495
  - Vascular lesion (vasc): F1=0.696, Prec=0.889, Rec=0.571, PR-AUC=0.854
  - Dermatofibroma (df): F1=0.429, Prec=0.750, Rec=0.300, PR-AUC=0.644

**Confusion matrix (rows=true, cols=predicted)**

| True \ Pred | nv | mel | bkl | bcc | akiec | vasc | df |
|-------------|----|-----|-----|-----|-------|------|----|
| nv | 645 | 9 | 8 | 4 | 3 | 1 | 0 |
| mel | 36 | 20 | 16 | 2 | 7 | 0 | 0 |
| bkl | 30 | 4 | 65 | 9 | 6 | 0 | 1 |
| bcc | 10 | 0 | 0 | 38 | 5 | 0 | 0 |
| akiec | 1 | 0 | 4 | 11 | 22 | 0 | 0 |
| vasc | 3 | 0 | 0 | 3 | 0 | 8 | 0 |
| df | 5 | 1 | 0 | 1 | 0 | 0 | 3 |

### Class-weighted CE
- Accuracy: 0.6585
- Balanced accuracy: 0.6373
- Macro F1: 0.5022

Per-class metrics:
  - Melanocytic nevus (nv): F1=0.796, Prec=0.962, Rec=0.679, PR-AUC=0.964
  - Melanoma (mel): F1=0.360, Prec=0.279, Rec=0.506, PR-AUC=0.302
  - Benign keratosis (bkl): F1=0.574, Prec=0.557, Rec=0.591, PR-AUC=0.657
  - Basal cell carcinoma (bcc): F1=0.528, Prec=0.376, Rec=0.887, PR-AUC=0.605
  - Actinic keratosis (akiec): F1=0.417, Prec=0.345, Rec=0.526, PR-AUC=0.484
  - Vascular lesion (vasc): F1=0.571, Prec=0.571, Rec=0.571, PR-AUC=0.677
  - Dermatofibroma (df): F1=0.269, Prec=0.167, Rec=0.700, PR-AUC=0.422

**Confusion matrix (rows=true, cols=predicted)**

| True \ Pred | nv | mel | bkl | bcc | akiec | vasc | df |
|-------------|----|-----|-----|-----|-------|------|----|
| nv | 455 | 95 | 35 | 38 | 12 | 4 | 31 |
| mel | 8 | 41 | 11 | 9 | 10 | 2 | 0 |
| bkl | 10 | 8 | 68 | 16 | 12 | 0 | 1 |
| bcc | 0 | 0 | 1 | 47 | 3 | 0 | 2 |
| akiec | 0 | 3 | 6 | 9 | 20 | 0 | 0 |
| vasc | 0 | 0 | 0 | 5 | 0 | 8 | 1 |
| df | 0 | 0 | 1 | 1 | 1 | 0 | 7 |

## Key observations

- Majority class NV is well recognized (F1 > 0.9 ordinary) but dominates accuracy.
- Minority classes MEL, AKIEC, DF are the weakest (F1 < 0.45 ordinary).
- MEL is frequently confused with NV (and to a lesser extent BKL/BCC).
- Class-weighted CE raises balanced accuracy (0.56 -> 0.64) but lowers overall accuracy (0.82 -> 0.66) and macro-F1 (0.60 -> 0.50), showing the head/tail trade-off.
- Weighted CE improves MEL recall (25% -> ~51%) but hurts NV precision substantially, shifting many NV predictions toward minority classes.