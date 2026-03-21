#!/usr/bin/env python3
"""
scripts/train_model.py

Train or retrain the performance classification model.
Uses synthetic data + any real session CSVs in data/sessions/.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --synthetic-only
    python scripts/train_model.py --n-synthetic 5000
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.trainer import PerformanceTrainer


def main():
    parser = argparse.ArgumentParser(description="Train NeuroRehab ML model")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Ignore real session data, use only synthetic")
    parser.add_argument("--n-synthetic", type=int, default=3000,
                        help="Number of synthetic training samples")
    args = parser.parse_args()

    print("\n🧠 NeuroRehab — Model Training")
    print("═" * 45)
    print(f"  Synthetic samples:  {args.n_synthetic}")
    print(f"  Use real data:      {not args.synthetic_only}")
    print()

    trainer = PerformanceTrainer()
    meta = trainer.train(
        use_real_data=not args.synthetic_only,
        n_synthetic=args.n_synthetic,
    )

    print("\n── Results ──────────────────────────────────")
    print(f"  Accuracy:    {meta.get('accuracy', 0):.3f}")
    print(f"  CV Accuracy: {meta.get('cv_accuracy', 0):.3f} ± {meta.get('cv_std', 0):.3f}")
    print(f"  N samples:   {meta.get('n_samples', 0)}")
    print(f"  Classes:     {meta.get('classes', [])}")
    print(f"\n  Model saved ✓")


if __name__ == "__main__":
    main()
