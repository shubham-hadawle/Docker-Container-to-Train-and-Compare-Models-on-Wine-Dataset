"""
Docker Lab 1 (modified): train a wine cultivar classifier inside a container.

Instead of fitting one fixed model, a few candidate pipelines are compared with
stratified k-fold cross validation on the training split. The best one is
refit, checked on a held-out test split and saved along with a metrics file
and a small HTML report.

All outputs go to ARTIFACT_DIR, so mount a volume there if you want to keep
them after the container exits.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from report import write_report

# read from env vars so they can be changed with `docker run -e ...`
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

MODEL_FILE = "wine_model.joblib"
CLASS_NAMES = ["cultivar_1", "cultivar_2", "cultivar_3"]


def build_candidates(seed):
    # same scaler in front of every model so the comparison is fair
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        # SVC(probability=True) is deprecated in sklearn 1.9, calibrate it instead
        "svm_rbf": CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }
    return {
        name: Pipeline([("scaler", StandardScaler()), ("model", model)])
        for name, model in models.items()
    }


def to_builtin(obj):
    # json can't handle numpy ints/floats directly
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"{type(obj)} is not JSON serializable")


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    wine = load_wine()
    X, y = wine.data, wine.target
    print(f"Wine dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(CLASS_NAMES)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train / test rows: {len(X_train)} / {len(X_test)}")

    # step 1: compare the candidates with CV (training data only)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates(RANDOM_STATE)
    cv_results = {}

    print(f"\n{CV_FOLDS}-fold cross validation (macro F1)")
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_results[name] = {
            "mean_f1": round(float(scores.mean()), 4),
            "std_f1": round(float(scores.std()), 4),
        }
        print(f"  {name:<22}{scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_results, key=lambda n: cv_results[n]["mean_f1"])
    print(f"\nSelected model: {best_name}")

    # step 2: refit the winner on the whole training split and score the test split
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "per_class": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
        ),
    }
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # step 3: save the model (with the info predict.py needs), metrics and report
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILE)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_names": list(wine.feature_names),
            "class_names": CLASS_NAMES,
        },
        model_path,
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "name": "UCI Wine",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "settings": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        "class_names": CLASS_NAMES,
        "cv_results": cv_results,
        "best_model": best_name,
        "test": test_metrics,
    }
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=to_builtin)

    report_path = os.path.join(ARTIFACT_DIR, "report.html")
    write_report(metrics, report_path)

    print("Model training was successful. Saved:")
    for path in (model_path, metrics_path, report_path):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
