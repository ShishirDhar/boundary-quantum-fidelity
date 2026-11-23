Based on your supervisor's instructions and the **Boundary-Aware Quantum Fidelity for Minority Recognition and OOD Rejection** proposal, I would treat this week as your **research familiarization + dataset exploration + classical baseline week**.

Your goal is not to implement the quantum model yet. You should first understand the problem, understand the data, establish a classical baseline, and identify what makes the problem difficult.

I'd break your work into four major tasks.

# **1. Understand the research and review related papers**



Priority 1

&#x20;Literature review



Don't try to read every paper in the bibliography equally. Start with the papers most closely related to your proposed research.

The proposal already identifies several important research directions.

**Recommended reading order:**

| Order | Paper / Method | What you should understand                                                                     |
| ----- | -------------- | ---------------------------------------------------------------------------------------------- |
| 1     | Q-SCOPE        | Quantum fidelity-based OOD detection. Probably the most important quantum paper to understand. |
| 2     | QuPID          | Input-dependent quantum circuits for medical image retrieval.                                  |
| 3     | OAMPL          | Classical open-set medical diagnosis using multiple prototypes.                                |
| 4     | OLPR           | Prototype separation and false-rejection correction in long-tailed recognition.                |
| 5     | PASCL          | OOD detection in long-tailed recognition.                                                      |
| 6     | AdaptOD        | Addressing head–tail bias in OOD detection.                                                    |
| 7     | LDAM           | Class-dependent margins for imbalanced classification.                                         |
| 8     | BPaCo          | Balanced contrastive learning for medical image classification.                                |

These are particularly important because they help establish what has already been done and what your proposed method might contribute. The proposal explicitly warns against claiming that margins, quantum fidelity, prototypes, or OOD rejection are individually novel.

Boundary_Guided_Quantum_Fidelity_Proposal.pdf

**For each paper, prepare a short research note containing:**

1. What problem does the paper address?
2. What dataset(s) does it use?
3. What architecture or methodology does it propose?
4. How does it handle class imbalance?
5. How does it handle OOD detection, if applicable?
6. What are its limitations?
7. What could your proposed approach learn from it?

Don't simply summarize the abstract. You should understand the methodology and experimental setup.

**Expected deliverable:** A literature review document containing approximately 1–2 pages per important paper.

# **2. Download and investigate HAM10000**



Priority 1

&#x20;Dataset exploration



The proposal identifies HAM10000 as the primary benchmark.

[DermAI 1.0: A Robust, Generalized, and Novel Attention-Enabled Ensemble-Based Transfer Learning Paradigm for Multiclass Classification of Skin Lesion Images](https://images.openai.com/static-rsc-4/cHxYniW40EreuPVmhxrh7LdPnaC_iMMarOATsu6ZKeUNYeCkrg2il-tZdfiGB7rgHPMdNRJSGhXnBSenk3f2Y3RnAk0uA53TYZSa3AFOhN28AW6hRFNeU5u7-IqT9EVmIZxpiOubBVoqsOegQxUYCqt3XahKnnpIQidTqRNlrzM?purpose=inline)

**HAM10000**

- 10,015 dermoscopic images.
- Seven diagnostic categories.
- Strongly imbalanced class distribution.
- Includes image metadata and lesion identifiers.
- Intended for your primary open-set experiment.

Download it from the [Harvard Dataverse](https://dataverse.harvard.edu/?utm_source=chatgpt.com) record referenced by the proposal.

You should investigate the following:

**A. Dataset structure**

- How are images stored?
- What does the metadata contain?
- How are image filenames associated with diagnostic labels?
- Are there patient IDs or lesion IDs?
- Are multiple images associated with the same lesion?

**B. Class distribution**

Generate a bar chart showing the number of images belonging to each diagnostic category.

The seven categories are:

| Label | Diagnosis                                     |
| ----- | --------------------------------------------- |
| NV    | Melanocytic nevus                             |
| MEL   | Melanoma                                      |
| BCC   | Basal cell carcinoma                          |
| AKIEC | Actinic keratoses / intraepithelial carcinoma |
| BKL   | Benign keratosis-like lesions                 |
| DF    | Dermatofibroma                                |
| VASC  | Vascular lesions                              |

Investigate:

- Which classes are dominant?
- Which classes are underrepresented?
- What is the majority-to-minority ratio?
- Are there significant differences in image characteristics?

**C. Visual exploration**

Create a grid of sample images from all seven classes.

Ideally, show 5–10 images per class.

This will help you understand whether the classes have visually distinguishable characteristics and whether there are potentially confusing categories.

**D. Metadata exploration**

Investigate available attributes such as:

- Age
- Sex
- Anatomical location
- Diagnosis
- Lesion identifier

Check for missing values and duplicates.

An important point: do not assume that 10,015 images represent 10,015 independent patients. The proposal requires grouping by patient when reliable identifiers are available, otherwise by lesion identifier.

Boundary_Guided_Quantum_Fidelity_Proposal.pdf

# **3. Train a simple classical neural network**



Priority 2

&#x20;Baseline experiment



This is probably the most important practical task this week.

Before implementing the hybrid quantum-classical model, establish how well a conventional neural network performs on the dataset.

I recommend starting with a pretrained CNN rather than training a CNN from scratch.

**Suggested baseline architecture**

**HAM10000 Images**

Input: 224 × 224 RGB

**Pretrained ResNet18**

Feature extraction / fine-tuning

**Fully Connected Layer**

Seven output neurons

**Softmax Classification**

Seven diagnostic classes

You can implement this using PyTorch.

**What experiments should you perform?**

Start with two configurations:

| Experiment | Training strategy               |
| ---------- | ------------------------------- |
| Baseline 1 | Ordinary cross-entropy training |
| Baseline 2 | Class-weighted cross-entropy    |

The first establishes how a conventional classifier behaves under imbalance. The second gives you an initial indication of whether class weighting helps minority recognition.

Don't spend too much time tuning hyperparameters. Your goal is to establish a reference point.

**Metrics to collect:**

- Accuracy
- Macro F1-score
- Per-class precision and recall
- Confusion matrix
- Per-class F1-score
- PR-AUC, particularly for minority classes

Accuracy alone is misleading for imbalanced datasets.

For example, a classifier that always predicts the majority class might achieve high accuracy while completely failing to recognize minority classes.

# **4. Generate visualizations and analyze the results**



Priority 2

&#x20;Experimental observations



Once your baseline is trained, generate the following visualizations.

[Machine Learning for Imbalanced Data | Data | Paperback](https://images.openai.com/static-rsc-4/YzRAvOIOA9rCtBrHSzuSvbaEHdoxlFUPKj4mAjD4mCjtkeidsqu8dttVw55YKoSPoxypD2TfvpeSogXW1a4uPMiDKKQk9oaxXI5_tO2kXwQbfuNg11eMMhMfVysaBl9RIOeTgHn5PDPaaLK86_2mfVv6ZI60O72j9C2HRKgOzw4?purpose=inline)

**1. Class distribution**

A bar chart showing the number of samples per diagnosis.

Establishes the degree of imbalance.

[A comprehensive survey of loss functions and metrics in deep learning | Artificial Intelligence Review | Springer Nature Link](https://images.openai.com/static-rsc-4/wGuwXuu4Tas_B-jdOx2Y-IrkrAMp6KWZtJ_Iof4rMrBkkIRe_5aygqfDNvgiI9FNDGMqragYqt5q1YQMFB2wdSHEP_WwLo3kKzUyc_jfOTQiQzp6h4hjaTpKm41wU0eCYBrssCZMjdbJSRJKtn9KzScvkK4-R2OlfGYDEahHGeQ?purpose=inline)

**2. Confusion matrix**

Determine which classes are confused with other classes.

Pay particular attention to minority-to-majority misclassification.

[Attention-Driven Feature Extraction for XAI in Histopathology: Leveraging a Hybrid Xception-CBAM Architecture for Multi-Cancer Diagnosis\[v1\] | Preprints.org](https://images.openai.com/static-rsc-4/Mrx60N25wLuJhnh9T0An8Qp7i1b0H8Tw0HW8qktbGF_YaqaPVa81sthK1liD-B1elJ5aWkhYttUkS3F6YNzA_wjUE20INQ89agEL5I6Rj_VyvCyEbxAOJO_lPHBHIEmziTsNkwGTwYVAY1b7vOcdCaRb46J-vayvmcJ6RmBYIyE?purpose=inline)

**3. Feature-space visualization**

Extract the neural network's penultimate-layer embeddings and visualize them using t-SNE or UMAP.

Observe whether different diagnostic categories form distinguishable clusters.

[plot_precision_recall_curve — scikit-plots 0.5.dev0 documentation](https://images.openai.com/static-rsc-4/OKSUl4dPE1WQA8S_SzxdxgwPOL1zQrCALCNIEX75V5E2dme2B3MMfAaqN6a2n3vrnnGGpxmtpBIwUASfsEVJKCAW9-8-k_f3j1AxQbLThIlJWmM8xDMrT6PsLRHIwgEXHOXxha-H1Ap9b6KnwoJfCxA7gpEdJLBHxtQYpVPUOMo?purpose=inline)

**4. Per-class performance**

Plot precision, recall and F1-score for each diagnosis.

Identify which classes the model struggles with.

The feature-space visualization is particularly relevant to your research.

Your initial hypothesis concerns whether quantum feature representations can produce more useful boundaries between majority and minority classes.

You should first investigate what happens in a conventional neural network's feature space.

However, remember that t-SNE/UMAP visualizations are exploratory: visually separated clusters do not prove that a classifier has learned reliable decision boundaries.

# **Your actual schedule for this week**

I'd organize the work into the following seven-day schedule.

## **Week 1 research roadmap**

1. **Day 1 — Understand the proposal**
   - Read Sections 1–4 carefully.
   - Understand the research questions.
   - Understand HAM10000 and the proposed open-set task.
   - Create a list of unfamiliar concepts.
2. **Day 2 — Literature review I**
   - Read Q-SCOPE.
   - Read QuPID.
   - Understand quantum fidelity-based classification and OOD detection.
   - Write research notes.
3. **Day 3 — Literature review II**
   - Read OAMPL, OLPR and PASCL.
   - Understand classical approaches to open-set recognition and class imbalance.
   - Identify their limitations.
4. **Day 4 — Download HAM10000**
   - Download images and metadata.
   - Inspect directory structure.
   - Investigate missing values and duplicate images.
   - Generate class distribution charts.
5. **Day 5 — Dataset visualization**
   - Display representative samples from all seven classes.
   - Investigate lesion identifiers.
   - Explore metadata.
   - Prepare a preliminary dataset report.
6. **Day 6 — Classical baseline**
   - Implement ResNet18.
   - Train using ordinary cross-entropy.
   - Evaluate per-class metrics.
   - Generate a confusion matrix.
7. **Day 7 — Analysis and documentation**
   - Try class-weighted cross-entropy.
   - Extract embeddings and generate t-SNE/UMAP.
   - Compare the two baselines.
   - Document observations and potential research gaps.

# **What should you have ready by the end of the week?**

I'd aim to produce these five deliverables:

**Weekly deliverables**

0/5 completed

**Literature review**

Notes on the most relevant quantum and classical papers.

**Dataset report**

HAM10000 structure, metadata, class distribution and sample visualizations.

**Dataset integrity report**

Duplicate checks, lesion/patient grouping and preliminary split manifest.

**Classical baseline**

Working ResNet18 implementation with ordinary and class-weighted training.

**Experimental analysis**

Confusion matrices, per-class metrics and feature-space visualization.

# **A few important things to keep in mind**

There are three things in the proposal that I particularly want you to understand before starting.

**1. Don't merge all minority classes into one class.**

The proposal treats majority and minority as frequency groups, not as two diagnostic labels. All four known diagnoses in a fold remain separate classification categories.

For example, Fold A uses:

| Role              | Classes             |
| ----------------- | ------------------- |
| Known ID          | NV, MEL, BCC, AKIEC |
| Auxiliary unknown | BKL                 |
| Final unknown     | DF, VASC            |

There are three such folds in the proposal. You should understand this setup before creating your final train/test splits.

**2. Don't accidentally introduce data leakage.**

Images belonging to the same patient or lesion must not appear across training and testing partitions. Grouping must happen before splitting, feature extraction or augmentation.

Also, the final unknown categories must never enter training or model selection.

**3. Don't start implementing the quantum circuit yet.**

Your immediate task is to understand the classical behavior. The proposal's later stages explicitly assign classical baselines before quantum circuit implementation and boundary-aware training.&#x20;

Boundary_Guided_Quantum_Fidelity_Proposal.pdf



## **My recommendation**

If I were approaching this research project, I would prioritize three questions throughout this week:

1. **Data:** How imbalanced is HAM10000 really, and how many independent lesion/patient groups are available per class?
2. **Classification:** Which minority classes does a conventional neural network confuse with majority classes, and why?
3. **Representation:** Do the learned feature embeddings show meaningful class separation, or are different diagnoses heavily overlapping?

These three questions will give you a much stronger foundation for understanding why the proposed quantum fidelity-based boundary learning might be useful.

**One final suggestion:** Keep your code, plots, experimental configurations and observations in a reproducible Git repository from day one. Your work this week should become the reference against which the subsequent hybrid classical-quantum experiments are compared.


## **1. ResNet18: train from scratch or use transfer learning?**

**I recommend transfer learning for your first baseline.** Train a ResNet18 initialized with ImageNet weights, then compare it with a model trained from scratch only if you have enough time and compute.

Your immediate goal is to establish a reliable classical lesion-classification baseline—not to prove that a CNN can learn visual features from zero. HAM10000 has about 10,015 images across seven imbalanced classes, which is relatively small for training a deep CNN from scratch.

### **Why transfer learning is the better starting point**

| Approach          | What happens                                                 | Advantages                                                     | Disadvantages                                         |
| ----------------- | ------------------------------------------------------------ | -------------------------------------------------------------- | ----------------------------------------------------- |
| From scratch      | All ResNet18 weights start randomly                          | No ImageNet feature initialization                             | More data- and compute-hungry; easier to overfit      |
| Transfer learning | Start with ImageNet-pretrained weights                       | Usually converges faster; good general-purpose visual features | ImageNet features are not specifically dermatological |
| Fine-tuning       | Start pretrained, then update some or all layers on HAM10000 | Adapts features to dermoscopic images                          | Needs careful validation and regularization           |

A good first experiment is:

1. Load ImageNet-pretrained ResNet18.
2. Replace its final fully connected layer with a 7-class layer.
3. Freeze the backbone and train the new classifier.
4. Unfreeze the later residual blocks and fine-tune with a smaller learning rate.
5. Optionally unfreeze the entire network and fine-tune further.

That gives you a practical baseline and a useful comparison between frozen features and adapted features.

### **Recommended PyTorch setup**

Use `torchvision`'s ResNet18 weights. The pretrained weights are generic ImageNet weights, not weights trained on HAM10000.

```
import torchimport torch.nn as nnfrom torchvision.models import resnet18, ResNet18_Weights# Load ImageNet-pretrained ResNet18weights = ResNet18_Weights.DEFAULTmodel = resnet18(weights=weights)# Replace the 1000-class ImageNet head with your 7 lesion classesnum_features = model.fc.in_featuresmodel.fc = nn.Linear(num_features, 7)device = "cuda" if torch.cuda.is_available() else "cpu"model = model.to(device)
```

**Stage 1 — train only the classification head**

```
for param in model.parameters():    param.requires_grad = Falsefor param in model.fc.parameters():    param.requires_grad = Trueoptimizer = torch.optim.AdamW(    model.fc.parameters(),    lr=1e-3,    weight_decay=1e-4)criterion = nn.CrossEntropyLoss()
```

Train for a few epochs, monitoring validation macro-F1 and per-class recall—not just accuracy.

**Stage 2 — fine-tune the last residual block**

```
for param in model.layer4.parameters():    param.requires_grad = Trueoptimizer = torch.optim.AdamW([    {"params": model.layer4.parameters(), "lr": 1e-4},    {"params": model.fc.parameters(), "lr": 5e-4},], weight_decay=1e-4)
```

If validation performance improves, you can experiment with unfreezing `layer3` too. Use a smaller learning rate for the backbone than for the classifier head.

### **Preprocessing**

ResNet18 expects three-channel images. A reasonable starting point:

- Resize/crop to `224 × 224`.
- Convert to RGB.
- Normalize using ImageNet mean and standard deviation.
- Apply modest random horizontal/vertical flips, rotations, and small color variations to training images only.

For example:

```
from torchvision import transformstrain_transform = transforms.Compose([    transforms.Resize((224, 224)),    transforms.RandomHorizontalFlip(),    transforms.RandomVerticalFlip(),    transforms.RandomRotation(15),    transforms.ColorJitter(        brightness=0.1,        contrast=0.1,        saturation=0.1    ),    transforms.ToTensor(),    transforms.Normalize(        mean=[0.485, 0.456, 0.406],        std=[0.229, 0.224, 0.225]    )])eval_transform = transforms.Compose([    transforms.Resize((224, 224)),    transforms.ToTensor(),    transforms.Normalize(        mean=[0.485, 0.456, 0.406],        std=[0.229, 0.224, 0.225]    )])
```

Avoid aggressive augmentation initially: it can change clinically meaningful color and texture characteristics.

### **Handling HAM10000's class imbalance**

Do not rely on accuracy. Start with ordinary cross-entropy, then compare it against a class-weighted version:

```
# Compute these counts from the TRAINING subset only.# Example: train_counts = [count_nv, count_mel, ..., count_vasc]counts = torch.tensor(train_counts, dtype=torch.float32)class_weights = counts.sum() / (len(counts) * counts)class_weights = class_weights.to(device)criterion = nn.CrossEntropyLoss(weight=class_weights)
```

Compare the two approaches rather than assuming weighted loss is better. A weighted sampler is another option, but I would not combine it with class-weighted loss in your first experiment.

### **What to report**

At minimum:

- Overall accuracy and balanced accuracy
- Macro-F1
- Per-class precision, recall and F1
- Confusion matrix
- One-vs-rest PR-AUC, particularly for minority classes

Save the predictions and class probabilities for every test image. You'll need them later for calibration and comparison with your fidelity-based model.

## **2. Are there existing lesion-classification models that were not trained on HAM10000?**

**Yes. There are some promising candidates.** However, distinguish between:

1. A model whose supervised training dataset did not contain HAM10000.
2. A model whose entire pretraining history did not contain HAM10000.

The second is much harder to establish. A model might be trained on another lesion dataset but initialized from a foundation model whose pretraining data are not fully disclosed.

Here are two candidates worth investigating.

### **Candidate A: PAD-UFES-20-trained model**

[A sample of each type of skin lesion present in PAD-UFES-20 dataset. | Download Scientific Diagram](https://images.openai.com/static-rsc-4/sgs2q88em41rCTxGyPVz2zNUmY5VBlz7zi2PoLtcCtSHr3K8ab7eSS6G7sIQO9G7GaqgzTyzNcjeA_scRuUmL9cRBw8uOUNADPr04bIcLnDT4iPMwNdKLlcUoq-92GIeoUeN6oIWXopQeVRxZYadUNQjqy8m50Vz8Uet9y0axvo?purpose=inline)

**PAD-UFES-20**

- 2,298 images
- 1,641 lesions from 1,373 patients
- Six diagnostic categories
- Smartphone clinical photographs rather than dermoscopic images

Source: PAD-UFES-20 dataset documentation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Datasets at Hugging Face+1



I found a model card for a ViT-based skin lesion model whose stated training dataset is PAD-UFES-20. It describes a model intended for six lesion categories, including melanoma, nevus, BCC and SCC.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://mediatum.ub.tum.de\&sz=32)

Vit Att Model



This is interesting because PAD-UFES-20 is a separate dataset from HAM10000.

**Limitation:** PAD-UFES-20 contains clinical smartphone photographs, whereas HAM10000 contains dermoscopic images. This domain difference makes it a useful, but difficult, external-generalization experiment.

### **Candidate B: Models trained on ISBI 2016/2017 and evaluated on PH2**

I also found a study that trained its models using ISBI 2016 and ISBI 2017 data, then used the PH2 dataset exclusively for testing.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://pmc.ncbi.nlm.nih.gov\&sz=32)

PubMed Central (PMC)



This gives you another possible source of models trained on data other than HAM10000.

However, you should inspect the exact training images and any pretrained initialization before claiming complete independence from HAM10000.

### **Important: Avoid assuming ISIC 2019 is independent**

This is particularly relevant to your project.

The official ISIC documentation explicitly says that the ISIC 2019 training collection includes:

- HAM10000
- BCN20000
- MSK images

Therefore, a model trained on the full ISIC 2019 training set is **not an independent HAM10000-trained-data comparison**.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://challenge.isic-archive.com\&sz=32)

challenge.isic-archive.com



## **3. What I would actually do for your research**

I would create three distinct experiments.

**Experiment 1 — ImageNet → HAM10000**

Start here

- ResNet18 pretrained on ImageNet.
- Fine-tune on HAM10000.
- Establish your conventional classification baseline.
- Evaluate class imbalance and per-class errors.

**Experiment 2 — External model → HAM10000**

- Obtain a model trained on PAD-UFES-20, if its checkpoint is available.
- Evaluate it on HAM10000 without fine-tuning.
- Investigate how the difference in image acquisition affects predictions.
- Report label compatibility and domain shift.

**Experiment 3 — External model → HAM10000 fine-tuning**

- Start from the PAD-UFES-20 model.
- Fine-tune on HAM10000.
- Compare against ImageNet → HAM10000.

This tests whether dermatology-specific initialization provides an advantage over generic ImageNet initialization.

For Experiment 2, don't force incompatible labels into a seven-class comparison. For example, PAD-UFES-20 has SCC, which is not one of HAM10000's seven categories, and its labels do not map perfectly to HAM10000 labels. Define a defensible common-label subset or use a separate binary task.

## **4. One important detail about your proposal**

Your proposal is not simply a seven-class classification project. It defines three folds with four known diagnoses, an auxiliary unknown diagnosis, and final unknown diagnoses.

So I would separate your work into two stages:

- **Initial CNN baseline:** Seven-class HAM10000 classification to understand the dataset, learn the training pipeline, and visualize the errors.
- **Research-protocol baseline:** Follow the proposal's known/unknown class roles for each fold. Train only on eligible known classes; keep auxiliary unknowns out of the primary model and selection process, and preserve final unknowns for untouched evaluation.&#x20;

  Boundary_Guided_Quantum_Fidelity_Proposal.pdf



This distinction matters because a conventional seven-class softmax classifier is not automatically an OOD detector.

## **5. My suggested next steps**

In this order:

1. Train ImageNet-pretrained ResNet18 on HAM10000.
2. Generate a confusion matrix and per-class metrics.
3. Extract the 512-dimensional penultimate-layer embeddings.
4. Visualize those embeddings using t-SNE or UMAP, coloring points by diagnosis.
5. Investigate the PAD-UFES-20 model checkpoint and its training provenance.
6. Implement the proposal's four-known-class baseline.

**The most interesting comparison for your research is probably ImageNet-pretrained ResNet18 versus a dermatology-trained encoder, with both evaluated under exactly the same HAM10000 splits.**

I can help you build the complete PyTorch training pipeline next, including group-aware data loading, checkpoint saving, metrics, confusion matrices, and embedding visualization.