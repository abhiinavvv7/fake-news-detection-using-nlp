"""Inspect PolyglotFakeFacts workbooks and create a normalized article CSV."""
from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_CSV = ROOT / "data" / "processed" / "articles.csv"
SUMMARY = ROOT / "reports" / "data_summary.json"


def key(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


ALIASES = {
    "label": {"label", "class", "target", "category", "authenticity"},
    "translated_text": {"englishtranslatedversion", "englishtranslation", "translatedtext", "englishtext", "translation"},
    "original_text": {"newsoriginaltext", "originaltext", "articletext", "content", "text", "newscontent"},
    "headline": {"newsheadline", "headline", "title", "newstitle"},
    "language": {"language", "originallanguage", "newslanguage", "lang"},
    "url": {"url", "link", "articleurl"},
    "domain": {"webdomain", "domain", "source", "publisher"},
    "keywords": {"keywords", "keyword"},
}


def find_column(columns, name):
    normalized = {key(c): c for c in columns}
    return next((normalized[a] for a in ALIASES[name] if a in normalized), None)


def inferred_label(path):
    name = path.stem.lower()
    if "fake" in name:
        return "fake"
    if "real" in name or "non_fake" in name or "nonfake" in name:
        return "real"
    return None


def read_workbook(path):
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    for sheet_name, frame in sheets.items():
        print(f"\n{path.name} / {sheet_name}: {frame.shape[0]} rows × {frame.shape[1]} columns")
        print("Columns:", list(frame.columns))
        for candidate in ("label",):
            col = find_column(frame.columns, candidate)
            if col:
                print(f"{col} counts:", frame[col].astype("string").value_counts(dropna=False).to_dict())
        print("Null counts:", frame.isna().sum().to_dict())
        yield frame


def main():
    files = sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No .xlsx files found. Download the v2.0 workbooks into {RAW_DIR}")
    named_classes = [p for p in files if inferred_label(p) is not None]
    # The release includes both class-wise files and additional split files. Prefer
    # the class-wise pair when present so the same records are not loaded twice.
    if any("fake" in p.stem.lower() for p in named_classes) and any("real" in p.stem.lower() or "non_fake" in p.stem.lower() for p in named_classes):
        files = named_classes

    normalized_frames = []
    summaries = []
    for path in files:
        for frame in read_workbook(path):
            summary = {"file": path.name, "rows": int(len(frame)), "columns": [str(c) for c in frame.columns]}
            label_col = find_column(frame.columns, "label")
            label = frame[label_col] if label_col else inferred_label(path)
            if isinstance(label, str):
                labels = pd.Series(label, index=frame.index)
            elif label is not None:
                labels = label
            else:
                labels = None
            if labels is None:
                summaries.append(summary)
                continue

            out = pd.DataFrame(index=frame.index)
            out["label"] = labels
            for name in ALIASES:
                col = find_column(frame.columns, name)
                if col and name != "label":
                    out[name] = frame[col]
            out["source_file"] = path.name
            normalized_frames.append(out.reset_index(drop=True))
            if label_col:
                summary["label_counts"] = frame[label_col].astype("string").value_counts(dropna=False).to_dict()
            else:
                summary["label_counts"] = {str(inferred_label(path)): int(len(frame))}
            summaries.append(summary)

    if not normalized_frames:
        raise SystemExit("No label column found and no filename containing Fake or Real. Review the printed schema.")

    data = pd.concat(normalized_frames, ignore_index=True)
    raw_label = data["label"].astype("string").str.strip().str.lower()
    label_map = {"fake": "fake", "false": "fake", "1": "fake", "non-fake": "real", "non_fake": "real", "real": "real", "true": "real", "0": "real"}
    data["label"] = raw_label.map(label_map)
    unknown = sorted(raw_label[data["label"].isna()].dropna().unique().tolist())
    if unknown:
        raise SystemExit(f"Unrecognized label values: {unknown}. Extend label_map after confirming their meaning.")

    text_col = data.get("translated_text", pd.Series(index=data.index, dtype="object")).fillna("").astype(str).str.strip()
    original = data.get("original_text", pd.Series(index=data.index, dtype="object")).fillna("").astype(str).str.strip()
    headline = data.get("headline", pd.Series(index=data.index, dtype="object")).fillna("").astype(str).str.strip()
    data["text"] = text_col.mask(text_col.eq(""), original).mask(lambda s: s.eq(""), headline)
    data["text_source"] = "headline"
    data.loc[original.ne(""), "text_source"] = "original_text"
    data.loc[text_col.ne(""), "text_source"] = "translated_text"
    data = data[data["text"].str.strip().ne("")].drop_duplicates(subset=["label", "text"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_CSV, index=False)
    summary = {
        "input_files": summaries,
        "normalized_rows": int(len(data)),
        "normalized_label_counts": data["label"].value_counts().to_dict(),
        "text_source_counts": data["text_source"].value_counts().to_dict(),
        "output": str(OUT_CSV.relative_to(ROOT)),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(data)} normalized rows to {OUT_CSV}")
    print("Labels:", data["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
