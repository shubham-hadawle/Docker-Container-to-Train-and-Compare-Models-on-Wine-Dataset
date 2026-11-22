"""
Loads the model saved by train.py and predicts the cultivar for new samples.

Usage:
    python predict.py                        # uses sample_input.json
    python predict.py --input my_wines.json  # a JSON object or a list of objects

Each object needs all 13 Wine features (see sample_input.json for the keys).
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


def load_samples(path, feature_names):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    for i, row in enumerate(data, start=1):
        missing = [name for name in feature_names if name not in row]
        if missing:
            sys.exit(f"Sample {i} is missing features: {', '.join(missing)}")

    return np.array([[row[name] for name in feature_names] for row in data], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Predict wine cultivar with the trained model")
    parser.add_argument("--input", default="sample_input.json", help="JSON file with samples")
    parser.add_argument(
        "--model",
        default=os.path.join(ARTIFACT_DIR, "wine_model.joblib"),
        help="path to the saved model",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(
            f"No model found at {args.model}. Run train.py first "
            "(and mount the same artifacts folder/volume)."
        )

    bundle = joblib.load(args.model)
    model = bundle["model"]
    class_names = bundle["class_names"]

    X = load_samples(args.input, bundle["feature_names"])
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print(f"Model: {bundle['model_name']}  |  samples: {len(X)}\n")
    for i, (pred, probs) in enumerate(zip(predictions, probabilities), start=1):
        detail = ", ".join(f"{c}={p:.2f}" for c, p in zip(class_names, probs))
        print(f"Sample {i}: {class_names[pred]}  ({detail})")


if __name__ == "__main__":
    main()
