"""
EDA minim pentru dataset-ul phishing (text + label).

Funcționalități:
- Citește CSV-ul standardizat (coloane: text, label) din data/raw/phishing_{subset}.csv
- Calculează statistici: număr rânduri, distribuție etichete, valori lipsă, duplicate, lungimi text
- Generează 2 grafice: histogramă lungime text, bar chart distribuție etichete
- Salvează un raport Markdown în docs/datasets/EDA_report_{subset}.md
- Opțional: actualizează secțiunea EDA din data/README.md între marcaje speciale

Rulare exemple:
    python -m src.preprocessing.eda --subset combined_reduced --update-data-readme
    python -m src.preprocessing.eda --input data/raw/phishing_combined.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "preprocessing_config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DOCS_DIR = PROJECT_ROOT / "docs" / "datasets"
PLOTS_DIR = DOCS_DIR / "plots"
DATA_README = PROJECT_ROOT / "data" / "README.md"


def read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    stats["rows"] = len(df)
    stats["cols"] = len(df.columns)
    stats["columns"] = list(df.columns)

    # valori lipsă
    na_pct = df.isna().mean().round(4)
    stats["na_pct"] = na_pct.to_dict()

    # duplicate după text
    if "text" in df.columns:
        stats["duplicates_by_text"] = int(df.duplicated(subset=["text"]).sum())
        # lungime text
        lengths = df["text"].astype(str).str.len()
        stats["length_min"] = int(lengths.min())
        stats["length_max"] = int(lengths.max())
        stats["length_mean"] = float(lengths.mean())
        stats["length_median"] = float(lengths.median())
        stats["length_q1"] = float(lengths.quantile(0.25))
        stats["length_q3"] = float(lengths.quantile(0.75))

    # distribuția etichetelor
    if "label" in df.columns:
        label_counts = df["label"].value_counts(dropna=False)
        label_ratio = (label_counts / len(df)).round(4)
        stats["label_counts"] = label_counts.to_dict()
        stats["label_ratio"] = label_ratio.to_dict()

    return stats


def make_plots(df: pd.DataFrame, subset: str) -> dict:
    out = {}
    # histogramă lungimi text
    if "text" in df.columns:
        lengths = df["text"].astype(str).str.len()
        plt.figure(figsize=(8, 4))
        sns.histplot(lengths, bins=50, kde=False)
        plt.title(f"Histograma lungimii textului ({subset})")
        plt.xlabel("Lungime (caractere)")
        plt.ylabel("Frecvență")
        hist_path = PLOTS_DIR / f"hist_length_{subset}.png"
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        out["hist_length"] = hist_path

    # bar chart distribuție etichete
    if "label" in df.columns:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=df["label"].astype(str))
        plt.title(f"Distribuția etichetelor ({subset})")
        plt.xlabel("Etichetă")
        plt.ylabel("Număr")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2.0, height), ha='center', va='bottom')
        labels_path = PLOTS_DIR / f"labels_{subset}.png"
        plt.tight_layout()
        plt.savefig(labels_path)
        plt.close()
        out["labels_bar"] = labels_path

    return out


def write_report(stats: dict, subset: str, plot_paths: dict) -> Path:
    report_path = DOCS_DIR / f"EDA_report_{subset}.md"
    lines = []
    lines.append(f"# Raport EDA – subset: {subset}\n")
    lines.append("\n## Rezumat\n")
    lines.append(f"- Rânduri: {stats.get('rows')} | Coloane: {stats.get('cols')}\n")
    if "duplicates_by_text" in stats:
        lines.append(f"- Duplicate după text: {stats['duplicates_by_text']}\n")
    if "label_counts" in stats:
        lines.append("- Distribuție etichete (count | ratio):\n")
        for k, v in stats["label_counts"].items():
            ratio = stats["label_ratio"].get(k, 0)
            lines.append(f"  - {k}: {v} | {ratio}\n")

    lines.append("\n## Valori lipsă (procent pe coloană)\n")
    for col, pct in stats.get("na_pct", {}).items():
        lines.append(f"- {col}: {pct}\n")

    if "length_mean" in stats:
        lines.append("\n## Lungimi text (caractere)\n")
        lines.append(
            f"- min: {stats['length_min']}, q1: {stats['length_q1']:.0f}, mediană: {stats['length_median']:.0f}, q3: {stats['length_q3']:.0f}, max: {stats['length_max']}, medie: {stats['length_mean']:.1f}\n"
        )

    if plot_paths:
        lines.append("\n## Grafice\n")
        if "hist_length" in plot_paths:
            rel = plot_paths["hist_length"].relative_to(PROJECT_ROOT)
            lines.append(f"- Histograma lungimii textului: ![]({rel.as_posix()})\n")
        if "labels_bar" in plot_paths:
            rel = plot_paths["labels_bar"].relative_to(PROJECT_ROOT)
            lines.append(f"- Distribuția etichetelor: ![]({rel.as_posix()})\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    return report_path


def update_data_readme(stats: dict, subset: str):
    """Inserează un rezumat EDA în data/README.md între marcaje speciale."""
    marker_start = "<!-- EDA_SUMMARY_START -->"
    marker_end = "<!-- EDA_SUMMARY_END -->"

    summary_lines = [
        marker_start + "\n",
        f"### Rezumat EDA (subset: {subset})\n",
        f"- Rânduri: {stats.get('rows')} | Coloane: {stats.get('cols')}\n",
        f"- Duplicate după text: {stats.get('duplicates_by_text', 0)}\n",
    ]
    if "label_counts" in stats:
        summary_lines.append("- Distribuție etichete:\n")
        for k, v in stats["label_counts"].items():
            ratio = stats["label_ratio"].get(k, 0)
            summary_lines.append(f"  - {k}: {v} ({ratio})\n")

    summary_lines.append(marker_end + "\n")
    summary = "".join(summary_lines)

    if not DATA_README.exists():
        return

    content = DATA_README.read_text(encoding="utf-8")
    if marker_start in content and marker_end in content:
        pre = content.split(marker_start)[0]
        post = content.split(marker_end)[1]
        new_content = pre + summary + post
    else:
        # adaugă la final
        new_content = content + "\n\n" + summary

    DATA_README.write_text(new_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="EDA pentru dataset phishing (text + label)")
    parser.add_argument("--subset", type=str, default=None, help="subset: combined_reduced/combined/email/sms/url/html")
    parser.add_argument("--input", type=str, default=None, help="Cale CSV alternativă (dacă nu folosim subset)")
    parser.add_argument("--update-data-readme", action="store_true", help="Actualizează secțiunea EDA în data/README.md")
    args = parser.parse_args()

    cfg = read_config(CONFIG_PATH)
    subset = args.subset or (cfg.get("dataset", {}).get("subset")) or "combined_reduced"

    if args.input:
        csv_path = Path(args.input)
    else:
        csv_path = RAW_DIR / f"phishing_{subset}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Nu am găsit fișierul CSV: {csv_path}. Rulează mai întâi scriptul de descărcare.")

    ensure_dirs()

    df = pd.read_csv(csv_path)
    # asigurăm existența coloanelor
    missing_cols = {c for c in ("text", "label") if c not in df.columns}
    if missing_cols:
        raise ValueError(f"Lipsesc coloane necesare: {missing_cols}. Expected: text, label")

    stats = compute_stats(df)
    plots = make_plots(df, subset)
    report_path = write_report(stats, subset, plots)
    print(f"[OK] Raport EDA salvat: {report_path}")

    if args.update_data_readme:
        update_data_readme(stats, subset)
        print("[OK] Secțiunea EDA din data/README.md a fost actualizată.")


if __name__ == "__main__":
    main()
