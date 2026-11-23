import os
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

BASE = Path("/home/cat/Stuff/Quantum Research/HAM10000")
META_CSV = BASE / "HAM10000_metadata.csv"
IMG_DIRS = [BASE / "HAM10000_images_part_1", BASE / "HAM10000_images_part_2"]
OUT_DIR = BASE / "resnet18_transfer"
RUNS_DIR = OUT_DIR / "runs"
PLOTS_DIR = OUT_DIR / "plots"
CKPT_DIR = OUT_DIR / "checkpoints"

CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FT = 5
LR_HEAD = 1e-3
LR_FT_BACKBONE = 1e-4
LR_FT_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224


def load_metadata():
    rows = []
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def image_path_for(image_id):
    for d in IMG_DIRS:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def _valid_image(path):
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def build_records():
    rows = load_metadata()
    records = []
    missing = 0
    corrupt = 0
    for row in rows:
        p = image_path_for(row["image_id"])
        if p is None:
            missing += 1
            continue
        if not _valid_image(p):
            corrupt += 1
            continue
        records.append({
            "image_id": row["image_id"],
            "lesion_id": row["lesion_id"],
            "dx": row["dx"],
            "path": str(p),
            "age": row["age"],
            "sex": row["sex"],
            "localization": row["localization"],
            "dataset": row["dataset"],
        })
    print(f"Loaded {len(records)} records, {missing} missing images, {corrupt} corrupt images")
    return records


class HAM10000Dataset(Dataset):
    def __init__(self, records, transform, class_to_idx):
        self.records = records
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[rec["dx"]]
        return img, label


def group_aware_split(records, test_size=0.2, val_size=0.1, seed=SEED):
    """Split grouped by lesion_id to avoid leakage across patient/lesion."""
    by_lesion = defaultdict(list)
    for rec in records:
        by_lesion[rec["lesion_id"]].append(rec)

    lesion_ids = list(by_lesion.keys())
    labels = [by_lesion[lid][0]["dx"] for lid in lesion_ids]

    train_val_ids, test_ids = train_test_split(
        lesion_ids, test_size=test_size, stratify=labels, random_state=seed
    )

    train_val_labels = [by_lesion[lid][0]["dx"] for lid in train_val_ids]
    val_frac = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_frac, stratify=train_val_labels, random_state=seed
    )

    def flatten(ids):
        out = []
        for lid in ids:
            out.extend(by_lesion[lid])
        return out

    return flatten(train_ids), flatten(val_ids), flatten(test_ids)


def make_transforms():
    train_t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_t, eval_t


def build_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    return model


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_layer4(model):
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True


def class_weights_from_counts(counts, device):
    counts = np.array(counts, dtype=np.float32)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    running = 0.0
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        running += loss.item() * imgs.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    per_class_prec = precision_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    pr_auc = []
    for i in range(NUM_CLASSES):
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        pr_auc.append(ap)

    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=class_names, zero_division=0, output_dict=True,
    )

    return {
        "loss": running / len(loader.dataset),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {class_names[i]: float(per_class_f1[i]) for i in range(NUM_CLASSES)},
        "per_class_precision": {class_names[i]: float(per_class_prec[i]) for i in range(NUM_CLASSES)},
        "per_class_recall": {class_names[i]: float(per_class_rec[i]) for i in range(NUM_CLASSES)},
        "per_class_pr_auc": {class_names[i]: float(pr_auc[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_confusion_matrix(cm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].plot(epochs, history["val_bal_acc"], label="val bal acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[2].plot(epochs, history["val_macro_f1"], label="val macro F1")
    axes[2].set_title("Macro F1"); axes[2].set_xlabel("Epoch"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(name, train_records, val_records, test_records,
                   use_class_weights, stage1_epochs, stage2_epochs,
                   device):
    print(f"\n=== Experiment: {name} | class_weights={use_class_weights} ===")

    train_t, eval_t = make_transforms()
    train_ds = HAM10000Dataset(train_records, train_t, CLASS_TO_IDX)
    val_ds = HAM10000Dataset(val_records, eval_t, CLASS_TO_IDX)
    test_ds = HAM10000Dataset(test_records, eval_t, CLASS_TO_IDX)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, persistent_workers=True)

    model = build_model().to(device)
    train_counts = [sum(1 for r in train_records if r["dx"] == c) for c in CLASS_NAMES]

    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_state = None

    criterion_eval = nn.CrossEntropyLoss()

    # ---- Stage 1: train head only ----
    freeze_backbone(model)
    if use_class_weights:
        weights = class_weights_from_counts(train_counts, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    for ep in range(1, stage1_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion_eval, device, CLASS_NAMES)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_bal_acc"].append(val_metrics["balanced_accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        print(f"[S1] epoch {ep}: train_loss={tr_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- Stage 2: fine-tune layer4 + head ----
    unfreeze_layer4(model)
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": LR_FT_BACKBONE},
        {"params": model.fc.parameters(), "lr": LR_FT_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    for ep in range(1, stage2_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion_eval, device, CLASS_NAMES)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_bal_acc"].append(val_metrics["balanced_accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        print(f"[S2] epoch {ep}: train_loss={tr_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion_eval, device, CLASS_NAMES)
    print(f"\nTEST [{name}]: acc={test_metrics['accuracy']:.4f} bal_acc={test_metrics['balanced_accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}")
    print("Per-class F1:", {k: round(v, 4) for k, v in test_metrics["per_class_f1"].items()})
    print("Per-class PR-AUC:", {k: round(v, 4) for k, v in test_metrics["per_class_pr_auc"].items()})

    # Save artifacts
    torch.save(model.state_dict(), run_dir / "model_best.pt")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "name": name,
            "class_weights": use_class_weights,
            "train_counts": dict(zip(CLASS_NAMES, train_counts)),
            "history": history,
            "test": test_metrics,
        }, f, indent=2)
    save_confusion_matrix(np.array(test_metrics["confusion_matrix"]), CLASS_NAMES,
                          f"Confusion Matrix - {name}", PLOTS_DIR / f"cm_{name}.png")
    plot_history(history, PLOTS_DIR / f"history_{name}.png")

    # Save predictions
    preds_path = run_dir / "test_predictions.csv"
    model.eval()
    y_true, y_pred, y_prob, ids = [], [], [], []
    with torch.no_grad():
        for recs, loader in [(test_records, test_loader)]:
            pass
    # Re-run loader aligned with records order is complex; instead iterate dataset directly
    test_ds_ordered = HAM10000Dataset(test_records, eval_t, CLASS_TO_IDX)
    ordered_loader = DataLoader(test_ds_ordered, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2)
    with torch.no_grad():
        for imgs, labels in ordered_loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
            y_prob.extend(probs.numpy())
            ids.extend([test_records[i]["image_id"] for i in range(len(labels))])

    pred_df = pd.DataFrame({
        "image_id": ids,
        "true_label": [CLASS_NAMES[int(t)] for t in y_true],
        "pred_label": [CLASS_NAMES[int(p)] for p in y_pred],
    })
    for i, c in enumerate(CLASS_NAMES):
        pred_df[f"prob_{c}"] = [row[i] for row in y_prob]
    pred_df.to_csv(preds_path, index=False)

    return test_metrics


def plot_class_distribution(train_counts, val_counts, test_counts):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(NUM_CLASSES)
    w = 0.25
    ax.bar(x - w, [train_counts.get(c, 0) for c in CLASS_NAMES], w, label="train")
    ax.bar(x, [val_counts.get(c, 0) for c in CLASS_NAMES], w, label="val")
    ax.bar(x + w, [test_counts.get(c, 0) for c in CLASS_NAMES], w, label="test")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Count"); ax.set_title("Class distribution across splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    records = build_records()
    train_rec, val_rec, test_rec = group_aware_split(records)

    def counts(recs):
        c = Counter(r["dx"] for r in recs)
        return {k: c.get(k, 0) for k in CLASS_NAMES}

    train_counts = counts(train_rec)
    val_counts = counts(val_rec)
    test_counts = counts(test_rec)
    print("Train:", train_counts)
    print("Val:", val_counts)
    print("Test:", test_counts)
    plot_class_distribution(train_counts, val_counts, test_counts)

    results = {}
    for use_cw in [False, True]:
        name = "weighted" if use_cw else "ordinary"
        m = run_experiment(
            name=name,
            train_records=train_rec,
            val_records=val_rec,
            test_records=test_rec,
            use_class_weights=use_cw,
            stage1_epochs=EPOCHS_HEAD,
            stage2_epochs=EPOCHS_FT,
            device=device,
        )
        results[name] = m

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for name, m in results.items():
        print(f"{name}: acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
              f"macro_f1={m['macro_f1']:.4f}")
    print(f"\nArtifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
