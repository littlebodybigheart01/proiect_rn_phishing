"""
Descărcare și export dataset Hugging Face: ealvaradob/phishing-dataset

Pași:
- încarcă config/preprocessing_config.yaml pentru a afla subsetul (implicit: combined_reduced)
- încearcă să încarce subsetul cu `datasets.load_dataset`
- detectează coloana de text și etichetă
- exportă CSV standardizat cu coloane: text, label în data/raw/

Rulare exemplu:
    python -m src.data_acquisition.download_hf_dataset --subset combined_reduced

Notă: Necesită pachetul `datasets` și conexiune la internet pentru prima rulare.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, List

import yaml
from datasets import load_dataset, DatasetDict, Dataset, get_dataset_config_names


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "preprocessing_config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def detect_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None


def pick_split(ds: DatasetDict | Dataset) -> Dataset:
    if isinstance(ds, Dataset):
        return ds
    # prefer train, apoi validation, apoi test, apoi primul disponibil
    for key in ("train", "validation", "test"):
        if key in ds:
            return ds[key]
    # fallback: primul
    first_key = next(iter(ds.keys()))
    return ds[first_key]


def main():
    parser = argparse.ArgumentParser(description="Download HF phishing dataset and export CSV")
    parser.add_argument("--subset", type=str, default=None, help="Subset/config nume: ex. combined_reduced, combined, email, sms, url, html")
    parser.add_argument("--out", type=str, default=None, help="Calea către fișierul CSV de ieșire")
    args = parser.parse_args()

    cfg = read_config(CONFIG_PATH)
    subset = args.subset or (cfg.get("dataset", {}).get("subset")) or "combined_reduced"

    print(f"[INFO] Încărcare dataset: ealvaradob/phishing-dataset | subset={subset}")
    # Unele dataset-uri Hugging Face folosesc "name"/config pentru subset
    # Încercăm cu subset ca config; dacă eșuează, încercăm fără.
    try:
        # Unele dataset-uri cer cod remote (v3+ datasets). Activăm explicit.
        ds_any = load_dataset("ealvaradob/phishing-dataset", subset, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Nu am putut încărca cu subset='{subset}' ca config: {e}\n[INFO] Încerc fără specificarea subsetului…")
        try:
            ds_any = load_dataset("ealvaradob/phishing-dataset", trust_remote_code=True)
        except Exception as e2:
            # Oferim o eroare mai utilă, listând config-urile disponibile (dacă se pot obține)
            try:
                configs = get_dataset_config_names("ealvaradob/phishing-dataset", trust_remote_code=True)
            except Exception:
                configs = []
            raise RuntimeError(
                "Nu am putut încărca datasetul Hugging Face 'ealvaradob/phishing-dataset'. "
                f"Încearcă unul dintre config-urile disponibile: {configs} și asigură-te că ai conexiune la internet.\n"
                f"Eroare inițială (cu subset='{subset}'): {e}\nEroare fallback (fără subset): {e2}"
            )

    ds = pick_split(ds_any)
    print(f"[INFO] Split selectat: {getattr(ds, 'split', 'unknown')} | nume configuratie: {subset}")

    # Conversie la pandas pentru procesare flexibilă
    df = ds.to_pandas()
    cols = list(df.columns)

    text_candidates = ["text", "message", "email_text", "sms_text", "content", "body"]
    label_candidates = ["label", "target", "class", "category"]

    text_col = detect_column(cols, [c.lower() for c in text_candidates])
    label_col = detect_column(cols, [c.lower() for c in label_candidates])

    if text_col is None:
        raise RuntimeError(f"Nu am reușit să detectez coloana de text. Coloane disponibile: {cols}")
    if label_col is None:
        raise RuntimeError(f"Nu am reușit să detectez coloana etichetă. Coloane disponibile: {cols}")

    # Standardizare nume coloane
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    # Asigurăm folderul de ieșire
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else RAW_DIR / f"phishing_{subset}.csv"

    # Export CSV
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Exportat: {out_path} | rânduri: {len(df)} | coloane: {list(df.columns)}")


if __name__ == "__main__":
    main()
