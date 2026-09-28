"""Compare TF-IDF + Logistic Regression with a TF-IDF + Linear SVM."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "articles.csv"
MODEL_DIR = ROOT / "models"
METRICS = ROOT / "reports" / "baseline_metrics.json"


def main():
    if not DATA.exists():
        raise SystemExit("Run `python src/inspect_data.py` first.")
    data = pd.read_csv(DATA).dropna(subset=["text", "label"])
    if data["label"].nunique() != 2:
        raise SystemExit(f"Expected two classes (fake/real); found {data['label'].value_counts().to_dict()}")
    counts = data["label"].value_counts()
    if counts.min() < 2:
        raise SystemExit("Need at least two examples of each class for a stratified holdout split.")

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"].astype(str), data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )
    labels = sorted(data["label"].unique())
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=42),
    }
    results = {}
    fitted_models = {}
    for name, classifier in classifiers.items():
        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=2, sublinear_tf=True)),
            ("classifier", classifier),
        ])
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        fitted_models[name] = model
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro")),
            "confusion_matrix_rows_true_columns_predicted": confusion_matrix(y_test, predicted, labels=labels).tolist(),
            "classification_report": classification_report(y_test, predicted, labels=labels, output_dict=True, zero_division=0),
        }

    winner = max(results, key=lambda name: results[name]["macro_f1"])
    metrics = {
        "dataset_rows": int(len(data)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "label_counts": counts.to_dict(),
        "labels": labels,
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "models": results,
        "best_model_by_macro_f1": winner,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    for name, model in fitted_models.items():
        joblib.dump(model, MODEL_DIR / f"tfidf_{name}.joblib")
    joblib.dump(fitted_models[winner], MODEL_DIR / "best_tfidf_model.joblib")
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Compared models on the same held-out rows ({len(x_test)} articles):")
    for name, result in results.items():
        print(f"{name}: accuracy={result['accuracy']:.4f}, macro F1={result['macro_f1']:.4f}")
    print(f"Best by macro F1: {winner}")
    print(f"Saved both fitted models and best model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
