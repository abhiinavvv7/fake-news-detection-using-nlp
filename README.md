# Fake News Detection Using NLP

A B.Tech project that prepares labeled news text from **PolyglotFakeFacts v2.0**, compares two TF-IDF classifiers, visualizes their scores in a Jupyter notebook, and provides a small interactive browser demo called **Paper Signals**.

> The classifier detects patterns in its training data. Its output and confidence are estimates, not a fact-check.

## What the project does

1. Reads the dataset's Excel workbooks and reports their sheets, columns, missing values, and available labels.
2. Normalizes article text and labels into `data/processed/articles.csv`.
3. Trains and compares TF-IDF + Logistic Regression with TF-IDF + Linear SVM using the same stratified 80/20 holdout split.
4. Saves evaluation metrics and model files, and displays charts in `model_comparison.ipynb`.
5. Runs a local web page where you can paste text and see the Logistic Regression model's predicted class and class probabilities.

## Project files

```text
├── app.py                      # Local web server and prediction endpoint
├── web/                        # Paper Signals page, styles, and interactions
├── data/
│   ├── raw/                    # Dataset Excel workbooks (place them here)
│   └── processed/              # articles.csv created by inspection
├── models/                     # Trained model pipelines
├── reports/                    # Dataset summary and model evaluation metrics
├── src/
│   ├── inspect_data.py         # Inspect workbooks and prepare articles.csv
│   └── train_baseline.py       # Train and compare baseline classifiers
├── model_comparison.ipynb      # Matplotlib results and score charts
├── requirements.txt
└── README.md
```

## 1. Set up Python

From this project folder, create and activate a virtual environment, then install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate it with `.venv\\Scripts\\activate` instead.

## 2. Add and inspect the dataset

Download **PolyglotFakeFacts v2.0** from [Mendeley Data](https://data.mendeley.com/datasets/gff8bmr4ff/2) and place its Excel workbook files in `data/raw/`.

Run:

```bash
python src/inspect_data.py
```

The script prints the workbook sheets, column names, row counts, missing-value counts, and label counts. It writes the normalized dataset to `data/processed/articles.csv` and a summary to `reports/data_summary.json`. When both `Fake.xlsx` and `Real.xlsx` are present, it uses those class-wise files rather than also loading split workbooks, to avoid duplicate records.

Text selection preference is English translation, original article text, then headline. Labels are normalized to `fake` and `real`; rows without usable text and exact same-label text duplicates are removed. URL and domain are preserved as metadata, not used as model text.

## 3. Train and compare the models

```bash
python src/train_baseline.py
```

This requires `data/processed/articles.csv`. It trains word unigram/bigram TF-IDF with Logistic Regression and Linear SVM on the same stratified 80/20 split (seed 42), then reports accuracy, macro F1, per-class metrics, and confusion matrices. It saves:

- `models/tfidf_logistic_regression.joblib`
- `models/tfidf_linear_svm.joblib`
- `models/best_tfidf_model.joblib` (the model with the higher macro F1)
- `reports/baseline_metrics.json`

The browser demo uses the Logistic Regression model because it provides class probabilities directly. The notebook reads the saved metrics file and visualizes the model comparison; train the models first if the metrics file is missing or stale.

## 4. View the charts in Jupyter

With the project environment activated, install/register the notebook kernel if needed:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name fake-news-nlp --display-name "Python (Fake News NLP)"
```

Open `model_comparison.ipynb`, select **Python (Fake News NLP)** as its kernel, and choose **Run All**. It displays Matplotlib charts for accuracy, macro F1, and confusion matrices using `reports/baseline_metrics.json`.

## 5. Run the interactive web demo

Train the models first so the Logistic Regression pipeline exists, then from the project folder with the environment active run:

```bash
python app.py
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). Paste at least five words and select **Read the signals** to see the predicted class and both class probabilities. **Load a demo text** fills the text box; **Clear** resets it. The page is served locally and the server binds to `127.0.0.1`. Stop it with Ctrl+C in the terminal.

If startup says the model is missing, run `python src/inspect_data.py` and `python src/train_baseline.py` first. If a port error says 8501 is already in use, stop the other copy of the app before restarting.

## Notes for interpreting results

This is a first academic baseline, not a reliable truth-verification system. Accuracy can be influenced by publisher, topic, translation, and dataset artifacts. Treat the app's confidence as the model's probability estimate; check the source, date, and evidence independently. Useful next evaluations include comparing translated and original text, reporting performance by language, and testing a source-disjoint split to measure publisher leakage.

## Dataset citation

Ciobanu, Alexandru (2026). *PolyglotFakeFacts: A multilingual dataset of fake and real news across politics, security, and social domains*, Version 2. Mendeley Data. [https://doi.org/10.17632/gff8bmr4ff.2](https://doi.org/10.17632/gff8bmr4ff.2). CC BY 4.0; include attribution when using the dataset in project reports.
