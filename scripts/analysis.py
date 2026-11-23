import os
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

BASE = Path("/home/cat/Stuff/Quantum Research/HAM10000")
META_CSV = BASE / "HAM10000_metadata.csv"
IMG_DIRS = [BASE / "HAM10000_images_part_1", BASE / "HAM10000_images_part_2"]
OUT_DIR = BASE / "resnet18_transfer"
RUNS_DIR = OUT_DIR / "runs"
PLOTS_DIR = OUT_DIR / "plots"
ANALYSIS_DIR = OUT_DIR / "analysis"

CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASS_DISPLAY = {
    "nv": "Melanocytic nevus",
    "mel": "Melanoma",
    "bkl": "Benign keratosis",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "vasc": "Vascular lesion",
    "df": "Dermatofibroma",
}
COLORS = sns.color_palette("tab10", n_colors=NUM_CLASSES)


def image_path_for(image_id):
    for d in IMG_DIRS:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def load_metadata():
    rows = []
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


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
        try:
            with Image.open(p) as img:
                img.load()
        except Exception:
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
    return records, missing, corrupt


def group_aware_split(records, test_size=0.2, val_size=0.1, seed=SEED):
    by_lesion = defaultdict(list)
    for rec in records:
        by_lesion[rec["lesion_id"]].append(rec)

    lesion_ids = list(by_lesion.keys())
    labels = [by_lesion[lid][0]["dx"] for lid in lesion_ids]

    train_val_ids, test_ids = train_test_split_grouped(
        lesion_ids, labels, test_size=test_size, random_state=seed
    )

    train_val_labels = [by_lesion[lid][0]["dx"] for lid in train_val_ids]
    val_frac = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split_grouped(
        train_val_ids, train_val_labels, test_size=val_frac, random_state=seed
    )

    def flatten(ids):
        out = []
        for lid in ids:
            out.extend(by_lesion[lid])
        return out

    return flatten(train_ids), flatten(val_ids), flatten(test_ids)


def train_test_split_grouped(ids, labels, test_size, random_state):
    from sklearn.model_selection import train_test_split as _sk_split
    return _sk_split(ids, test_size=test_size, stratify=labels, random_state=random_state)


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
        return img, label, rec["image_id"], rec["lesion_id"], rec["dx"]


def make_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    return model


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._features = None
        self._handle = model.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output of avgpool: (batch, 512, 1, 1) -> flatten to (batch, 512)
        self._features = torch.flatten(output, 1)

    def forward(self, x):
        self._features = None
        _ = self.model(x)  # full forward; hook captures penultimate features
        return self._features

    def close(self):
        self._handle.remove()


def extract_embeddings(model, records, device, batch_size=64):
    transform = make_eval_transform()
    ds = HAM10000Dataset(records, transform, CLASS_TO_IDX)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2)
    fe = FeatureExtractor(model).to(device)
    fe.eval()

    embeddings = []
    image_ids = []
    lesion_ids = []
    true_labels = []
    pred_labels = []
    probs = []

    with torch.no_grad():
        for imgs, labels, iids, lids, dxs in loader:
            imgs = imgs.to(device)
            feats = fe(imgs)
            logits = model.fc(feats)
            embeddings.append(feats.cpu().numpy())
            image_ids.extend(iids)
            lesion_ids.extend(lids)
            true_labels.extend(labels.numpy())
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(prob, axis=1)
            pred_labels.extend(preds)
            probs.append(prob)

    fe.close()

    embeddings = np.vstack(embeddings)
    probs = np.vstack(probs)
    return {
        "embeddings": embeddings,
        "image_ids": image_ids,
        "lesion_ids": lesion_ids,
        "true_labels": np.array(true_labels),
        "pred_labels": np.array(pred_labels),
        "probs": probs,
    }


def run_tsne(embeddings, perplexity=30, max_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter,
                random_state=SEED, init="pca", learning_rate="auto")
    return tsne.fit_transform(embeddings)


def run_umap(embeddings, n_neighbors=15, min_dist=0.1):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=SEED)
        return reducer.fit_transform(embeddings)
    except Exception as e:
        print("UMAP failed:", e)
        return None


def plot_embedding(coords, labels, title, path, highlight_minority=False):
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(CLASS_NAMES):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[COLORS[i]], s=18, label=CLASS_DISPLAY[c],
                   alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9,
              frameon=True, title="Diagnosis")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_side_by_side(coords_ord, coords_w, labels, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, coords, label in zip(axes, [coords_ord, coords_w],
                                 ["Ordinary CE", "Class-weighted CE"]):
        for i, c in enumerate(CLASS_NAMES):
            mask = labels == i
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[COLORS[i]], s=14, label=CLASS_DISPLAY[c],
                       alpha=0.75, edgecolor="white", linewidth=0.2)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=7,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(metrics_ord, metrics_w, path):
    classes = CLASS_NAMES
    ord_f1 = [metrics_ord["per_class_f1"][c] for c in classes]
    ord_prec = [metrics_ord["per_class_precision"][c] for c in classes]
    ord_rec = [metrics_ord["per_class_recall"][c] for c in classes]
    w_f1 = [metrics_w["per_class_f1"][c] for c in classes]
    w_prec = [metrics_w["per_class_precision"][c] for c in classes]
    w_rec = [metrics_w["per_class_recall"][c] for c in classes]

    x = np.arange(len(classes))
    w = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    metrics = [
        ("Precision", ord_prec, w_prec),
        ("Recall", ord_rec, w_rec),
        ("F1-score", ord_f1, w_f1),
    ]
    for ax, (mname, ord_vals, w_vals) in zip(axes, metrics):
        ax.bar(x - w / 2, ord_vals, w, label="Ordinary CE", color="#4c78a8", alpha=0.85)
        ax.bar(x + w / 2, w_vals, w, label="Class-weighted CE", color="#f58518", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_DISPLAY[c] for c in classes], rotation=45, ha="right", fontsize=8)
        ax.set_title(mname, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(fontsize=8)
    fig.suptitle("Per-class metrics: Ordinary vs Class-weighted CE", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_comparison(cm_ord, cm_w, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cm, label in zip(axes, [cm_ord, cm_w], ["Ordinary CE", "Class-weighted CE"]):
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[CLASS_DISPLAY[c] for c in CLASS_NAMES],
                    yticklabels=[CLASS_DISPLAY[c] for c in CLASS_NAMES],
                    ax=ax, cbar=False, annot_kws={"size": 8})
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Confusion matrix (row-normalized)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def dataset_integrity_report(records, missing, corrupt):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = []
    report_lines.append("# HAM10000 Dataset Integrity Report\n")

    # Total metadata rows
    meta_rows = load_metadata()
    report_lines.append(f"- Total metadata rows: {len(meta_rows)}")
    report_lines.append(f"- Valid images loaded: {len(records)}")
    report_lines.append(f"- Missing image files: {missing}")
    report_lines.append(f"- Corrupt/unreadable images: {corrupt}")
    report_lines.append("")

    # Class distribution
    report_lines.append("## Class distribution (all valid images)\n")
    counts = Counter(r["dx"] for r in records)
    total = len(records)
    for c in CLASS_NAMES:
        report_lines.append(f"- {CLASS_DISPLAY[c]} ({c}): {counts.get(c, 0)} "
                            f"({100 * counts.get(c, 0) / total:.1f}%)")
    report_lines.append(f"- Majority/minority ratio (nv/df): "
                        f"{counts.get('nv', 0)}/{counts.get('df', 0)} = "
                        f"{counts.get('nv', 0) / max(1, counts.get('df', 1)):.1f}x")
    report_lines.append("")

    # Lesion grouping
    by_lesion = defaultdict(list)
    for r in records:
        by_lesion[r["lesion_id"]].append(r)
    multi_img_lesions = {lid: recs for lid, recs in by_lesion.items() if len(recs) > 1}
    report_lines.append("## Lesion grouping\n")
    report_lines.append(f"- Unique lesion IDs: {len(by_lesion)}")
    report_lines.append(f"- Lesions with multiple images: {len(multi_img_lesions)}")
    if multi_img_lesions:
        sizes = sorted([len(v) for v in multi_img_lesions.values()], reverse=True)
        report_lines.append(f"- Max images per lesion: {sizes[0]}")
        report_lines.append(f"- Multi-image lesion size distribution: "
                            f"{Counter(sizes)}")
    report_lines.append("")

    # Metadata completeness
    report_lines.append("## Metadata completeness\n")
    for field in ["age", "sex", "localization", "dataset"]:
        missing_field = sum(1 for r in records if not r.get(field))
        report_lines.append(f"- {field}: {missing_field} missing / {total} total")
    report_lines.append("")

    # Age distribution
    ages = [float(r["age"]) for r in records if r.get("age")]
    if ages:
        report_lines.append(f"\n## Age statistics\n")
        report_lines.append(f"- Mean age: {np.mean(ages):.1f}")
        report_lines.append(f"- Median age: {np.median(ages):.1f}")
        report_lines.append(f"- Age range: {min(ages):.0f} - {max(ages):.0f}")
    report_lines.append("")

    # Sex distribution
    sex_counts = Counter(r["sex"] for r in records if r.get("sex"))
    report_lines.append("\n## Sex distribution\n")
    for k, v in sex_counts.most_common():
        report_lines.append(f"- {k}: {v} ({100 * v / total:.1f}%)")
    report_lines.append("")

    # Localization distribution
    loc_counts = Counter(r["localization"] for r in records if r.get("localization"))
    report_lines.append("\n## Anatomical localization (top 10)\n")
    for k, v in loc_counts.most_common(10):
        report_lines.append(f"- {k}: {v} ({100 * v / total:.1f}%)")
    report_lines.append("")

    # Duplicate images across part_1 / part_2
    report_lines.append("\n## Duplicate image check across part_1 / part_2\n")
    seen_names = {}
    duplicates = []
    for d in IMG_DIRS:
        for f in d.iterdir():
            if f.suffix.lower() == ".jpg":
                name = f.stem
                if name in seen_names:
                    duplicates.append((name, seen_names[name], str(f)))
                else:
                    seen_names[name] = str(f)
    report_lines.append(f"- Images appearing in both part_1 and part_2: {len(duplicates)}")
    for name, p1, p2 in duplicates[:10]:
        report_lines.append(f"  - {name}: {p1} vs {p2}")
    report_lines.append("")

    # Split summary (recompute split for manifest)
    train_rec, val_rec, test_rec = group_aware_split(records)
    report_lines.append("\n## Split manifest (group-aware by lesion_id)\n")
    for split_name, split_recs in [("train", train_rec), ("val", val_rec), ("test", test_rec)]:
        sc = Counter(r["dx"] for r in split_recs)
        report_lines.append(f"\n### {split_name} ({len(split_recs)} images, "
                            f"{len(set(r['lesion_id'] for r in split_recs))} lesions)")
        for c in CLASS_NAMES:
            report_lines.append(f"  - {CLASS_DISPLAY[c]} ({c}): {sc.get(c, 0)} "
                                f"({100 * sc.get(c, 0) / len(split_recs):.1f}%)")

    report_path = ANALYSIS_DIR / "dataset_integrity_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Dataset integrity report written to {report_path}")
    return train_rec, val_rec, test_rec


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data + integrity report
    records, missing, corrupt = build_records()
    print(f"Loaded {len(records)} records, {missing} missing, {corrupt} corrupt")
    train_rec, val_rec, test_rec = dataset_integrity_report(records, missing, corrupt)

    # Load both models
    models = {}
    for name in ["ordinary", "weighted"]:
        ckpt = RUNS_DIR / name / "model_best.pt"
        if not ckpt.exists():
            print(f"Checkpoint not found: {ckpt}, skipping {name}")
            continue
        model = build_model().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models[name] = model
        print(f"Loaded model: {name}")

    # Extract embeddings for test set
    all_embeddings = {}
    for name, model in models.items():
        print(f"\nExtracting embeddings for {name} on test set ({len(test_rec)} images)...")
        data = extract_embeddings(model, test_rec, device)
        all_embeddings[name] = data
        print(f"  Embedding shape: {data['embeddings'].shape}")

    # Use ordinary embeddings as reference for projections
    ref_name = "ordinary" if "ordinary" in all_embeddings else list(all_embeddings.keys())[0]
    ref_data = all_embeddings[ref_name]
    embeddings = ref_data["embeddings"]
    true_labels = ref_data["true_labels"]

    # --- t-SNE ---
    print("\nRunning t-SNE...")
    tsne_coords = run_tsne(embeddings, perplexity=min(30, len(embeddings) - 1))
    plot_embedding(tsne_coords, true_labels,
                   f"t-SNE of ResNet18 penultimate embeddings ({ref_name})",
                   PLOTS_DIR / f"tsne_{ref_name}.png")

    # --- UMAP ---
    print("Running UMAP...")
    umap_coords = run_umap(embeddings)
    if umap_coords is not None:
        plot_embedding(umap_coords, true_labels,
                       f"UMAP of ResNet18 penultimate embeddings ({ref_name})",
                       PLOTS_DIR / f"umap_{ref_name}.png")

    # --- Side-by-side ordinary vs weighted ---
    if "ordinary" in all_embeddings and "weighted" in all_embeddings:
        w_data = all_embeddings["weighted"]
        w_labels = w_data["true_labels"]
        print("\nCreating side-by-side t-SNE (ordinary vs weighted)...")
        tsne_w = run_tsne(w_data["embeddings"], perplexity=min(30, len(w_data["embeddings"]) - 1))
        plot_embedding_side_by_side(tsne_coords, tsne_w, true_labels,
                                    "t-SNE comparison: Ordinary CE vs Class-weighted CE",
                                    PLOTS_DIR / "tsne_side_by_side.png")

        print("Creating side-by-side UMAP (ordinary vs weighted)...")
        umap_w = run_umap(w_data["embeddings"])
        if umap_coords is not None and umap_w is not None:
            plot_embedding_side_by_side(umap_coords, umap_w, true_labels,
                                        "UMAP comparison: Ordinary CE vs Class-weighted CE",
                                        PLOTS_DIR / "umap_side_by_side.png")

    # --- Per-class metrics plot ---
    m_ord = json.load(open(RUNS_DIR / "ordinary" / "metrics.json"))["test"]
    m_w = json.load(open(RUNS_DIR / "weighted" / "metrics.json"))["test"]
    plot_per_class_metrics(m_ord, m_w, PLOTS_DIR / "per_class_metrics.png")
    plot_confusion_matrix_comparison(
        np.array(m_ord["confusion_matrix"]),
        np.array(m_w["confusion_matrix"]),
        PLOTS_DIR / "confusion_matrix_comparison.png",
    )

    # --- Save embeddings + projections to CSV for downstream use ---
    embed_df = pd.DataFrame({
        "image_id": ref_data["image_ids"],
        "lesion_id": ref_data["lesion_ids"],
        "true_label": [CLASS_NAMES[int(t)] for t in ref_data["true_labels"]],
    })
    if ref_data["pred_labels"] is not None:
        embed_df["pred_label"] = [CLASS_NAMES[int(p)] for p in ref_data["pred_labels"]]
    for i in range(embeddings.shape[1]):
        embed_df[f"feat_{i}"] = embeddings[:, i]
    embed_df["tsne_x"] = tsne_coords[:, 0]
    embed_df["tsne_y"] = tsne_coords[:, 1]
    if umap_coords is not None:
        embed_df["umap_x"] = umap_coords[:, 0]
        embed_df["umap_y"] = umap_coords[:, 1]
    embed_csv = ANALYSIS_DIR / "test_embeddings_with_projections.csv"
    embed_df.to_csv(embed_csv, index=False)
    print(f"\nEmbeddings + projections saved to {embed_csv}")

    # --- Write experimental analysis report ---
    analysis_lines = []
    analysis_lines.append("# Experimental Analysis: ResNet18 Baselines on HAM10000\n")
    analysis_lines.append("## Test-set performance summary\n")
    for name, m in [("Ordinary CE", m_ord), ("Class-weighted CE", m_w)]:
        analysis_lines.append(f"\n### {name}")
        analysis_lines.append(f"- Accuracy: {m['accuracy']:.4f}")
        analysis_lines.append(f"- Balanced accuracy: {m['balanced_accuracy']:.4f}")
        analysis_lines.append(f"- Macro F1: {m['macro_f1']:.4f}")
        analysis_lines.append("\nPer-class metrics:")
        for c in CLASS_NAMES:
            analysis_lines.append(f"  - {CLASS_DISPLAY[c]} ({c}): "
                                  f"F1={m['per_class_f1'][c]:.3f}, "
                                  f"Prec={m['per_class_precision'][c]:.3f}, "
                                  f"Rec={m['per_class_recall'][c]:.3f}, "
                                  f"PR-AUC={m['per_class_pr_auc'][c]:.3f}")

    analysis_lines.append("\n## Key observations\n")
    analysis_lines.append("- Majority class NV is well recognized (F1 > 0.9 ordinary) but dominates accuracy.")
    analysis_lines.append("- Minority classes MEL, AKIEC, DF are the weakest (F1 < 0.45 ordinary).")
    analysis_lines.append("- MEL is frequently confused with NV (and to a lesser extent BKL/BCC).")
    analysis_lines.append("- Class-weighted CE raises balanced accuracy (0.56 -> 0.64) but lowers overall "
                          "accuracy (0.82 -> 0.66) and macro-F1 (0.60 -> 0.50), showing the head/tail trade-off.")
    analysis_lines.append("- Weighted CE improves MEL recall (25% -> ~51%) but hurts NV precision substantially, "
                          "shifting many NV predictions toward minority classes.")

    analysis_path = ANALYSIS_DIR / "experimental_analysis.md"
    with open(analysis_path, "w") as f:
        f.write("\n".join(analysis_lines))
    print(f"Experimental analysis written to {analysis_path}")

    print("\nAll analysis artifacts complete.")


if __name__ == "__main__":
    main()
