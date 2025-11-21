"""
Preprocesare + split pentru dataset text (coloane: text, label), conform config/preprocessing_config.yaml.

Pași implementați (config-driven):
- Curățare/transformări:
  - strip_html (BeautifulSoup) → text
  - normalize_whitespace
  - lowercase (opțional)
  - remove_urls (opțional; regex simplu)
  - length_clip (opțional; taie la max_chars)
  - remove_duplicates (pe coloana text)
  - drop_empty_text (după transformări)

- Split: train/validation/test conform raportelor din config, stratificat pe label (dacă e setat), cu seed/shuffle.

Output:
- CSV procesat: data/processed/processed_{subset}.csv
- CSV-uri pentru split: data/train/train_{subset}.csv, data/validation/validation_{subset}.csv, data/test/test_{subset}.csv
- Raport scurt (opțional): docs/datasets/Preprocess_report_{subset}.md

Rulare exemple:
    python -m src.preprocessing.preprocess_and_split \
      --input data/raw/phishing_combined_reduced.csv \
      --config config/preprocessing_config.yaml

    # Sau pe baza subsetului din config (implicit):
    python -m src.preprocessing.preprocess_and_split --subset combined_reduced
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "preprocessing_config.yaml"


def read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def detect_subset_from_input(input_path: Path) -> str:
    """Încearcă să deducă subsetul din numele fișierului, ex: phishing_combined_reduced.csv → combined_reduced.
    Dacă nu reușește, returnează "custom".
    """
    name = input_path.stem
    if name.startswith("phishing_") and len(name.split("_", 1)) == 2:
        return name.split("_", 1)[1]
    return "custom"


def strip_html(text: str) -> str:
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return text


URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?",
    flags=re.IGNORECASE,
)


def remove_urls(text: str) -> str:
    # Heuristic simplu: elimină URL-urile evidente. Pentru phishing păstrăm URL-urile (config implicit false).
    return URL_PATTERN.sub("", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class PreprocessStats:
    initial_rows: int = 0
    after_drop_empty: int = 0
    after_dedup: int = 0
    clipped_rows: int = 0
    removed_empty_count: int = 0
    removed_duplicates_count: int = 0


def preprocess_df(df: pd.DataFrame, cfg: dict, stats: PreprocessStats) -> pd.DataFrame:
    pp = cfg.get("preprocessing", {})
    missing = cfg.get("missing_values", {})
    outliers = cfg.get("outliers", {})

    # Asigură coloanele necesare
    missing_cols = {c for c in ("text", "label") if c not in df.columns}
    if missing_cols:
        raise ValueError(f"Lipsesc coloane necesare: {missing_cols}. Expected: text, label")

    stats.initial_rows = len(df)

    # Transformări pe text
    def transform_text(t: str) -> str:
        if not isinstance(t, str):
            t = "" if pd.isna(t) else str(t)
        if pp.get("strip_html", False):
            t = strip_html(t)
        if pp.get("remove_urls", False):
            t = remove_urls(t)
        if pp.get("normalize_whitespace", False):
            t = normalize_whitespace(t)
        if pp.get("lowercase", False):
            t = t.lower()
        return t

    df["text"] = df["text"].apply(transform_text)

    # Length clip
    clip_cfg = outliers.get("length_clip", {})
    clipped_rows = 0
    if clip_cfg.get("enabled", False):
        max_chars = int(clip_cfg.get("max_chars", 2000))
        lengths = df["text"].astype(str).str.len()
        clipped_rows = int((lengths > max_chars).sum())
        df.loc[lengths > max_chars, "text"] = df.loc[lengths > max_chars, "text"].astype(str).str[:max_chars]
    stats.clipped_rows = clipped_rows

    # Drop empty text
    if missing.get("drop_empty_text", False):
        before = len(df)
        df = df[df["text"].astype(str).str.strip() != ""].copy()
        stats.after_drop_empty = len(df)
        stats.removed_empty_count = before - stats.after_drop_empty
    else:
        stats.after_drop_empty = len(df)

    # Remove duplicates by text
    if pp.get("remove_duplicates", False):
        before = len(df)
        df = df.drop_duplicates(subset=["text"]).copy()
        stats.after_dedup = len(df)
        stats.removed_duplicates_count = before - stats.after_dedup
    else:
        stats.after_dedup = len(df)

    return df


def do_split(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.8))
    val_ratio = float(split_cfg.get("validation", 0.1))
    test_ratio = float(split_cfg.get("test", 0.1))
    stratify_flag = bool(split_cfg.get("stratify", True))
    seed = int(split_cfg.get("random_seed", 42))
    shuffle = bool(split_cfg.get("shuffle", True))

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Raporturile de split nu însumează 1.0")

    y = df["label"] if stratify_flag else None

    # Întâi separăm train vs temp (val+test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df, train_y, temp_y = train_test_split(
        df, y, test_size=temp_ratio, random_state=seed, shuffle=shuffle, stratify=y
    )

    # Apoi împărțim temp în val și test proporțional
    val_share_of_temp = val_ratio / temp_ratio if temp_ratio > 0 else 0
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_share_of_temp),
        random_state=seed,
        shuffle=shuffle,
        stratify=temp_y if stratify_flag else None,
    )

    return train_df, val_df, test_df


def write_report(stats: PreprocessStats, subset: str, cfg: dict, shapes: Tuple[int, int, int], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"Preprocess_report_{subset}.md"
    train_n, val_n, test_n = shapes
    lines = [
        f"# Raport Preprocesare – subset: {subset}\n\n",
        "## Rezumat\n",
        f"- Rânduri inițiale: {stats.initial_rows}\n",
        f"- După eliminare texte goale: {stats.after_drop_empty} (eliminate: {stats.removed_empty_count})\n",
        f"- După eliminare duplicate: {stats.after_dedup} (eliminate: {stats.removed_duplicates_count})\n",
        f"- Rânduri tăiate la max_chars (clip): {stats.clipped_rows}\n\n",
        "## Split\n",
        f"- Train: {train_n}\n",
        f"- Validation: {val_n}\n",
        f"- Test: {test_n}\n\n",
        "## Config folosit (rezumat)\n",
        f"- lowercase: {cfg.get('preprocessing', {}).get('lowercase', False)}\n",
        f"- strip_html: {cfg.get('preprocessing', {}).get('strip_html', False)}\n",
        f"- remove_urls: {cfg.get('preprocessing', {}).get('remove_urls', False)}\n",
        f"- normalize_whitespace: {cfg.get('preprocessing', {}).get('normalize_whitespace', False)}\n",
        f"- remove_duplicates: {cfg.get('preprocessing', {}).get('remove_duplicates', True)}\n",
        f"- split ratios (train/val/test): {cfg.get('split', {}).get('train', 0.8)}/"
        f"{cfg.get('split', {}).get('validation', 0.1)}/"
        f"{cfg.get('split', {}).get('test', 0.1)}\n",
        f"- stratify: {cfg.get('split', {}).get('stratify', True)}\n",
        f"- seed: {cfg.get('split', {}).get('random_seed', 42)}\n",
    ]
    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Preprocesare + split pentru dataset phishing (text + label)")
    parser.add_argument("--input", type=str, default=None, help="Cale CSV de input (dacă lipsește, se folosește subset → data/raw/phishing_{subset}.csv)")
    parser.add_argument("--subset", type=str, default=None, help="Numele subsetului (ex. combined_reduced). Folosit la denumiri output dacă nu e dedus din numele fișierului.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Calea spre config YAML")
    args = parser.parse_args()

    cfg = read_config(Path(args.config))

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Nu am găsit fișierul de input: {input_path}")
        subset = args.subset or detect_subset_from_input(input_path)
    else:
        subset = args.subset or (cfg.get("dataset", {}).get("subset")) or "combined_reduced"
        input_path = PROJECT_ROOT / "data" / "raw" / f"phishing_{subset}.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Nu am găsit fișierul CSV: {input_path}. Furnizează --input sau rulează scriptul de download/conversie.")

    # Directoare output
    save_paths = cfg.get("save_paths", {})
    processed_dir = PROJECT_ROOT / save_paths.get("processed_dir", "data/processed")
    train_dir = PROJECT_ROOT / save_paths.get("train_dir", "data/train")
    val_dir = PROJECT_ROOT / save_paths.get("validation_dir", "data/validation")
    test_dir = PROJECT_ROOT / save_paths.get("test_dir", "data/test")
    for d in (processed_dir, train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Logging/report
    logging_cfg = cfg.get("logging", {})
    save_reports = bool(logging_cfg.get("save_reports", True))
    reports_dir = PROJECT_ROOT / logging_cfg.get("reports_dir", "docs/datasets")

    print(f"[INFO] Citire CSV: {input_path}")
    df = pd.read_csv(input_path)

    stats = PreprocessStats()
    df_proc = preprocess_df(df, cfg, stats)

    # Salvează CSV procesat complet
    processed_path = processed_dir / f"processed_{subset}.csv"
    df_proc.to_csv(processed_path, index=False)
    print(f"[OK] Date procesate: {processed_path} | rânduri: {len(df_proc)}")

    # Split
    train_df, val_df, test_df = do_split(df_proc, cfg)

    train_path = train_dir / f"train_{subset}.csv"
    val_path = val_dir / f"validation_{subset}.csv"
    test_path = test_dir / f"test_{subset}.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Train: {train_path} | rânduri: {len(train_df)}")
    print(f"[OK] Validation: {val_path} | rânduri: {len(val_df)}")
    print(f"[OK] Test: {test_path} | rânduri: {len(test_df)}")

    if save_reports:
        report_path = write_report(stats, subset, cfg, (len(train_df), len(val_df), len(test_df)), reports_dir)
        print(f"[OK] Raport preprocesare: {report_path}")


if __name__ == "__main__":
    main()
