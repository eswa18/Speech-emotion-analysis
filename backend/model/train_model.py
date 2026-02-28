"""
Multi-dataset training script for Speech Emotion Recognition.
Supports: RAVDESS, TESS, CREMA-D (individually or combined).
Produces: trained model, confusion matrices, dataset comparison.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extraction import extract_features, extract_features_with_augmentation
from utils.dataset_loader import load_ravdess, load_tess, load_cremad, load_all_datasets
from model.model_pytorch import get_model
import config


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════
# Feature Extraction from File List
# ═══════════════════════════════════════════════════════════════

def extract_all_features(file_list, label_list, use_augmentation=True, fixed_length=200):
    """Extract features from a list of audio files."""
    features, labels = [], []
    total = len(file_list)

    for i, (filepath, label) in enumerate(zip(file_list, label_list)):
        try:
            if use_augmentation:
                aug_features = extract_features_with_augmentation(
                    filepath, fixed_length=fixed_length
                )
                for feat in aug_features:
                    features.append(feat)
                    labels.append(label)
            else:
                feat = extract_features(filepath, fixed_length=fixed_length)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"    Processed {i + 1}/{total} files...")
        except Exception as e:
            continue

    if len(features) == 0:
        return None, None
    return np.array(features), np.array(labels)


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_test, y_test, dataset_name="combined",
                epochs=60, lr=0.001, patience=10):
    """Train the CNN-BiLSTM model and return metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on {device} | Dataset: {dataset_name}")
    print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    train_dataset = EmotionDataset(X_train, y_train)
    test_dataset = EmotionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=len(config.EMOTIONS))
    class_counts = np.maximum(class_counts, 1)  # avoid division by zero
    weights = 1.0 / class_counts.astype(float)
    weights = weights / weights.sum() * len(config.EMOTIONS)
    class_weights = torch.FloatTensor(weights).to(device)

    input_size = X_train.shape[1]  # feature dimension (141)
    model = get_model(num_classes=len(config.EMOTIONS), input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=5,
                                                      verbose=True)

    best_acc = 0
    no_improve = 0
    model_save_path = os.path.join(os.path.dirname(__file__), "emotion_model.pth")

    print(f"\n🎬 Starting Training ({epochs} epochs, patience={patience})")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  🌟 Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f} (Best! Saved)")
        else:
            no_improve += 1
            if (epoch + 1) % 5 == 0:
                print(f"     Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | "
                      f"Val Acc: {val_acc:.4f}")

        # Early stopping
        if no_improve >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\n✅ Training Finished. Best Val Accuracy: {best_acc:.4f}")

    # ── Final Evaluation ──
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    final_preds, final_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_targets.extend(targets.cpu().numpy())

    print(f"\n📊 Classification Report ({dataset_name}):")
    report = classification_report(final_targets, final_preds,
                                    target_names=config.EMOTIONS, output_dict=True)
    print(classification_report(final_targets, final_preds,
                                 target_names=config.EMOTIONS))

    # Confusion Matrix
    cm = confusion_matrix(final_targets, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix — {dataset_name}')
    cm_path = os.path.join(os.path.dirname(__file__), f"confusion_matrix_{dataset_name.lower()}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  🖼️  Saved confusion matrix: {cm_path}")

    return {
        'dataset': dataset_name,
        'accuracy': round(best_acc * 100, 2),
        'report': report,
        'confusion_matrix': cm.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train SER Model")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['ravdess', 'tess', 'cremad', 'all'],
                        help='Dataset to train on (default: all)')
    parser.add_argument('--compare', action='store_true',
                        help='Train on each dataset separately and compare')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    args = parser.parse_args()

    use_augmentation = not args.no_augment
    results = []

    if args.compare:
        # ── Compare Mode: train separately on each dataset ──
        datasets = {
            'RAVDESS': (load_ravdess, config.RAVDESS_DIR),
            'TESS': (load_tess, config.TESS_DIR),
            'CREMA-D': (load_cremad, config.CREMAD_DIR),
        }

        for name, (loader, path) in datasets.items():
            if not os.path.exists(path):
                print(f"\n⚠️  {name} not found at {path}. Skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"  Training on {name}")
            print(f"{'='*60}")

            files, labels = loader(path)
            if len(files) == 0:
                print(f"  ❌ No files found. Skipping.")
                continue

            print(f"  📦 Extracting features...")
            X, y = extract_all_features(files, labels,
                                         use_augmentation=use_augmentation)
            if X is None:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            result = train_model(X_train, y_train, X_test, y_test,
                                  dataset_name=name, epochs=args.epochs)
            results.append(result)

        # Print comparison table
        if results:
            print_comparison(results)

    else:
        # ── Single / Combined training ──
        if args.dataset == 'all':
            files, labels, info = load_all_datasets(
                config.RAVDESS_DIR, config.TESS_DIR, config.CREMAD_DIR
            )
        elif args.dataset == 'ravdess':
            files, labels = load_ravdess(config.RAVDESS_DIR)
        elif args.dataset == 'tess':
            files, labels = load_tess(config.TESS_DIR)
        elif args.dataset == 'cremad':
            files, labels = load_cremad(config.CREMAD_DIR)

        if len(files) == 0:
            print("❌ No data found. Check dataset paths in config.py.")
            return

        print(f"\n📦 Extracting features (augmentation={'ON' if use_augmentation else 'OFF'})...")
        X, y = extract_all_features(files, labels, use_augmentation=use_augmentation)
        if X is None:
            print("❌ Feature extraction failed.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        result = train_model(X_train, y_train, X_test, y_test,
                              dataset_name=args.dataset.upper(), epochs=args.epochs)
        results.append(result)

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_path}")


def print_comparison(results):
    """Print a comparison table of dataset results."""
    print("\n" + "=" * 70)
    print("  📊 DATASET COMPARISON")
    print("=" * 70)
    print(f"  {'Dataset':<12} | {'Accuracy':>10} | {'Angry F1':>10} | {'Happy F1':>10} | "
          f"{'Neutral F1':>10} | {'Sad F1':>10}")
    print("-" * 70)

    for r in results:
        report = r['report']
        angry_f1 = report.get('Angry', {}).get('f1-score', 0)
        happy_f1 = report.get('Happy', {}).get('f1-score', 0)
        neutral_f1 = report.get('Neutral', {}).get('f1-score', 0)
        sad_f1 = report.get('Sad', {}).get('f1-score', 0)

        print(f"  {r['dataset']:<12} | {r['accuracy']:>9.2f}% | "
              f"{angry_f1:>9.3f} | {happy_f1:>9.3f} | "
              f"{neutral_f1:>9.3f} | {sad_f1:>9.3f}")

    print("=" * 70)


if __name__ == '__main__':
    main()
