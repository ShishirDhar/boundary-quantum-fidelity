# HAM10000 Dataset Integrity Report

- Total metadata rows: 10015
- Valid images loaded: 4846
- Missing image files: 3991
- Corrupt/unreadable images: 1178

## Class distribution (all valid images)

- Melanocytic nevus (nv): 3324 (68.6%)
- Melanoma (mel): 423 (8.7%)
- Benign keratosis (bkl): 545 (11.2%)
- Basal cell carcinoma (bcc): 260 (5.4%)
- Actinic keratosis (akiec): 177 (3.7%)
- Vascular lesion (vasc): 64 (1.3%)
- Dermatofibroma (df): 53 (1.1%)
- Majority/minority ratio (nv/df): 3324/53 = 62.7x

## Lesion grouping

- Unique lesion IDs: 4246
- Lesions with multiple images: 549
- Max images per lesion: 4
- Multi-image lesion size distribution: Counter({2: 501, 3: 45, 4: 3})

## Metadata completeness

- age: 3 missing / 4846 total
- sex: 0 missing / 4846 total
- localization: 0 missing / 4846 total
- dataset: 0 missing / 4846 total


## Age statistics

- Mean age: 52.0
- Median age: 50.0
- Age range: 0 - 85


## Sex distribution

- male: 2637 (54.4%)
- female: 2209 (45.6%)


## Anatomical localization (top 10)

- lower extremity: 1070 (22.1%)
- back: 1008 (20.8%)
- trunk: 750 (15.5%)
- upper extremity: 521 (10.8%)
- abdomen: 518 (10.7%)
- face: 380 (7.8%)
- chest: 163 (3.4%)
- foot: 156 (3.2%)
- neck: 82 (1.7%)
- scalp: 62 (1.3%)


## Duplicate image check across part_1 / part_2

- Images appearing in both part_1 and part_2: 0


## Split manifest (group-aware by lesion_id)


### train (3383 images, 2971 lesions)
  - Melanocytic nevus (nv): 2328 (68.8%)
  - Melanoma (mel): 298 (8.8%)
  - Benign keratosis (bkl): 374 (11.1%)
  - Basal cell carcinoma (bcc): 178 (5.3%)
  - Actinic keratosis (akiec): 122 (3.6%)
  - Vascular lesion (vasc): 44 (1.3%)
  - Dermatofibroma (df): 39 (1.2%)

### val (482 images, 425 lesions)
  - Melanocytic nevus (nv): 326 (67.6%)
  - Melanoma (mel): 44 (9.1%)
  - Benign keratosis (bkl): 56 (11.6%)
  - Basal cell carcinoma (bcc): 29 (6.0%)
  - Actinic keratosis (akiec): 17 (3.5%)
  - Vascular lesion (vasc): 6 (1.2%)
  - Dermatofibroma (df): 4 (0.8%)

### test (981 images, 850 lesions)
  - Melanocytic nevus (nv): 670 (68.3%)
  - Melanoma (mel): 81 (8.3%)
  - Benign keratosis (bkl): 115 (11.7%)
  - Basal cell carcinoma (bcc): 53 (5.4%)
  - Actinic keratosis (akiec): 38 (3.9%)
  - Vascular lesion (vasc): 14 (1.4%)
  - Dermatofibroma (df): 10 (1.0%)