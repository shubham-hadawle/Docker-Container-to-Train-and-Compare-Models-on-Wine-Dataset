# Docker Lab 1: Wine Cultivar Classifier in a Container

This is my version of [Docker Lab 1](https://github.com/shubham-hadawle/MLOps/tree/main/Labs/Docker_Labs/Lab1).

The original lab builds an image that trains a Random Forest on the Iris dataset, saves it as
`iris_model.pkl` with joblib and prints a success message. The image is then exported to a tar
file with `docker save`.

I kept the same workflow (build an image, train the model when the container runs, save it with
joblib, export the image with `docker save`) but changed the dataset, the training logic and how
the outputs are handled.

## What's different

| | Original Lab 1 | This version |
|---|---|---|
| Dataset | Iris (150 rows, 4 features) | Wine (178 rows, 13 chemical features, 3 cultivars) |
| Model | One Random Forest | 4 pipelines (StandardScaler + Logistic Regression / RBF SVM / Random Forest / Gradient Boosting), best one picked with stratified k-fold CV |
| Evaluation | None | Held-out test split: accuracy, macro F1, confusion matrix, per-class precision/recall |
| Outputs | `iris_model.pkl`, stays inside the container | `wine_model.joblib`, `metrics.json` and `report.html`, written to a mounted folder |
| Inference | None | `predict.py` loads the saved model and predicts new samples from a JSON file |
| Config | Hard coded | Environment variables: `TEST_SIZE`, `RANDOM_STATE`, `CV_FOLDS`, `ARTIFACT_DIR` |
| Base image | `python:3.10` | `python:3.12-slim` with pinned package versions |
| Dockerfile | Copies everything, then installs | Requirements copied first (layer caching), runs as a non-root user, `.dockerignore` |
| Running | `docker run` | `docker run`, plus a `docker-compose.yml` that runs training and then prediction |

## Project structure

```
Docker-Container-to-Train-and-Compare-Models-on-Wine-Dataset/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── artifacts/              # model, metrics and report are saved here (mounted into the container)
└── src/
    ├── train.py            # loads data, compares models, evaluates, saves everything
    ├── report.py           # builds report.html from the metrics
    ├── predict.py          # loads the saved model and predicts new samples
    ├── sample_input.json   # three example wines for predict.py
    └── requirements.txt
```

## How the training works

1. `train.py` loads the Wine dataset and makes a stratified 80/20 train/test split.
2. Every candidate pipeline is scored with stratified 5-fold cross validation (macro F1) on the
   training split only, so the test split is not used to choose the model.
3. The best pipeline is refit on the full training split and evaluated once on the test split.
4. The fitted pipeline is saved with joblib together with the feature names and class names, so
   `predict.py` can check that the input has the right columns.
5. `metrics.json` and `report.html` are written next to the model.

The SVM is wrapped in `CalibratedClassifierCV` because `SVC(probability=True)` is deprecated in
scikit-learn 1.9 and `predict.py` needs class probabilities.

## Running it

You need Docker Desktop (or Docker Engine) installed. Run the commands from this folder.
Tested with Docker 29.9 and Docker Compose v5.7 on an Apple Silicon Mac.

### 1. Build the image

```bash
docker build -t wine-lab1:v1 .
```

### 2. Train the model

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" wine-lab1:v1
```

The `-v` flag mounts the local `artifacts/` folder into the container, so the model, metrics and
report are still there after the container is removed. `docker run wine-lab1:v1` without it also
works (same as the original lab), but then the outputs only exist inside that container.

### 3. Predict with the saved model

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" wine-lab1:v1 python predict.py
```

To try your own samples, put a JSON file (same keys as `src/sample_input.json`) in `artifacts/`:

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" wine-lab1:v1 python predict.py --input /app/artifacts/my_wines.json
```

### 4. Change the training settings

```bash
docker run --rm -e CV_FOLDS=10 -e TEST_SIZE=0.25 -v "$(pwd)/artifacts:/app/artifacts" wine-lab1:v1
```

### 5. Save and load the image

```bash
docker save wine-lab1:v1 > wine_lab1_image.tar
```

```bash
docker load < wine_lab1_image.tar
```

The tar file is about 117 MB, so it is in `.gitignore`.

### 6. Or run both steps with Docker Compose

```bash
docker compose up --build
```

Compose builds the image, runs the `trainer` service and only starts `predictor` once training
has exited successfully (`condition: service_completed_successfully`). Remove the containers
afterwards with:

```bash
docker compose down
```

### 7. Open the report

```bash
open artifacts/report.html
```

(`xdg-open` on Linux, `start` on Windows.)

**Linux note:** the container runs as `mluser` (uid 1000). If you get a permission error when it
writes to `artifacts/`, add `--user "$(id -u):$(id -g)"` to the `docker run` command.

## Expected output

The seeds are fixed, so the numbers should be the same on every run with the default settings.

Training:

```
Wine dataset: 178 samples, 13 features, 3 classes
Train / test rows: 142 / 36

5-fold cross validation (macro F1)
  logistic_regression   0.9796 (+/- 0.0273)
  svm_rbf               0.9865 (+/- 0.0167)
  random_forest         0.9789 (+/- 0.0285)
  gradient_boosting     0.9259 (+/- 0.1000)

Selected model: svm_rbf
Test accuracy: 0.9722
Test macro F1: 0.9710

              precision    recall  f1-score   support

  cultivar_1       1.00      1.00      1.00        12
  cultivar_2       0.93      1.00      0.97        14
  cultivar_3       1.00      0.90      0.95        10

    accuracy                           0.97        36
   macro avg       0.98      0.97      0.97        36
weighted avg       0.97      0.97      0.97        36

Model training was successful. Saved:
  - /app/artifacts/wine_model.joblib
  - /app/artifacts/metrics.json
  - /app/artifacts/report.html
```

Prediction:

```
Model: svm_rbf  |  samples: 3

Sample 1: cultivar_1  (cultivar_1=0.98, cultivar_2=0.02, cultivar_3=0.00)
Sample 2: cultivar_2  (cultivar_1=0.00, cultivar_2=0.94, cultivar_3=0.06)
Sample 3: cultivar_3  (cultivar_1=0.00, cultivar_2=0.02, cultivar_3=0.98)
```

`report.html` shows the selected model and test scores at the top, followed by the
cross-validation comparison, the confusion matrix and the per-class scores.

## Cleaning up

```bash
docker rmi wine-lab1:v1
```
