# README – Proiect RN: Analiza și Pregătirea Datelor (Etapa 3)

Acesta este SINGURUL README al proiectului (documentația unificată). Orice alte fișiere README din subfoldere sunt considerate secundare și pot fi ignorate.

—

Disciplina: Rețele Neuronale  
Instituție: POLITEHNICA București – FIIR  
Student: Chelu Fabian-Cătălin  
Data: 2025-11-21

—

## 1) Prezentare pe scurt
Etapa 3 pregătește setul de date pentru antrenarea unui model de Rețele Neuronale: colectare/încărcare, analiză exploratorie (EDA), preprocesare și împărțire în train/validation/test. Pipeline-ul este reproductibil, configurabil prin `config/preprocessing_config.yaml`, iar rezultatele sunt salvate în `data/` și documentate în `docs/datasets/`.

—

## 2) Cerințe de sistem
- Python 3.9+ (recomandat 3.10/3.11)
- Dependențe Python:
```
pip install -r requirements.txt
```

Opțional (recomandat): mediu virtual
- Linux/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```
- Windows (PowerShell):
```
py -m venv .venv
.venv\Scripts\Activate.ps1
```

—

## 3) Rulare rapidă (3 pași)
Presupunem că datasetul CSV standardizat există la `data/raw/phishing_combined_reduced.csv`. Dacă ai doar JSON local, vezi Pasul 0.

1. Instalează dependențele
```
pip install -r requirements.txt
```

2. Rulează EDA și actualizează documentația
```
python -m src.preprocessing.eda --subset combined_reduced --update-data-readme
```
Rezultate: `docs/datasets/EDA_report_combined_reduced.md` + grafice în `docs/datasets/plots/` + rezumat EDA în `data/README.md`.

3. Preprocesare + Split (80/10/10, stratificat)
```
python -m src.preprocessing.preprocess_and_split --subset combined_reduced
```
Rezultate: CSV-uri în `data/processed/`, `data/train/`, `data/validation/`, `data/test/` + raport `docs/datasets/Preprocess_report_combined_reduced.md`.

—

## 0) Dacă ai JSON local și vrei să îl convertești în CSV
Pune fișierul la `data/raw/combined_reduced.json` (sau în rădăcina proiectului) și rulează:
```
python - << 'PY'
import json, pandas as pd
from pathlib import Path
src = Path('data/raw/combined_reduced.json')
if not src.exists():
    src = Path('combined_reduced.json')
print('[INFO] Citire:', src)
with open(src, 'r', encoding='utf-8') as f:
    data = json.load(f)
pd.DataFrame(data)[['text','label']].to_csv('data/raw/phishing_combined_reduced.csv', index=False)
print('[OK] Scris CSV: data/raw/phishing_combined_reduced.csv | rânduri:', len(data))
PY
```

—

## 4) Alternativă: descarcă direct de pe Hugging Face (opțional)
Script: `src/data_acquisition/download_hf_dataset.py`

Exemple:
```
python -m src.data_acquisition.download_hf_dataset --subset combined_reduced
# rezultat: data/raw/phishing_combined_reduced.csv
```
Notă pentru `datasets>=3`: dacă apare eroarea „Dataset scripts are no longer supported…”, scriptul folosește `trust_remote_code=True`. Dacă totuși este blocat, poți forța și prin variabilă de mediu:
- Linux/macOS:
```
export HF_DATASETS_TRUST_REMOTE_CODE=1
```
- Windows (PowerShell):
```
$env:HF_DATASETS_TRUST_REMOTE_CODE="1"
```

Listare config-uri disponibile (dacă subsetul nu e corect):
```
python - << 'PY'
from datasets import get_dataset_config_names
print(get_dataset_config_names('ealvaradob/phishing-dataset', trust_remote_code=True))
PY
```

—

## 5) Analiza Exploratorie a Datelor (EDA)
Script: `src/preprocessing/eda.py`

Rulare:
```
python -m src.preprocessing.eda --subset combined_reduced --update-data-readme
# sau explicit:
python -m src.preprocessing.eda --input data/raw/phishing_combined_reduced.csv --update-data-readme
```

Ce face:
- calculează statistici (rânduri, lipsuri, duplicate pe text, lungimi text)
- distribuția etichetelor și grafice
- scrie raportul în `docs/datasets/EDA_report_{subset}.md`
- inserează rezumat în `data/README.md` între marcaje speciale

Rezumat EDA actual (subset: combined_reduced):
- Rânduri: 77677 | Coloane: 2
- Duplicate după text: 109
- Distribuție etichete: 0 → 44975 (0.579), 1 → 32702 (0.421)

—

## 6) Preprocesare + Split (80/10/10, stratificat)
Script: `src/preprocessing/preprocess_and_split.py`

Rulare (folosind subset):
```
python -m src.preprocessing.preprocess_and_split --subset combined_reduced
```
Sau cu input explicit și config:
```
python -m src.preprocessing.preprocess_and_split \
  --input data/raw/phishing_combined_reduced.csv \
  --config config/preprocessing_config.yaml
```

Operații efectuate (controlate prin config):
- strip HTML → text (BeautifulSoup)
- normalize whitespace
- lowercase (opțional)
- remove URLs (opțional – implicit off pentru phishing)
- length clip (opțional)
- remove duplicates (pe `text`)
- drop empty text
- split stratificat: train/validation/test = 0.8/0.1/0.1, seed=42

Output așteptat:
- `data/processed/processed_{subset}.csv`
- `data/train/train_{subset}.csv`
- `data/validation/validation_{subset}.csv`
- `data/test/test_{subset}.csv`
- `docs/datasets/Preprocess_report_{subset}.md`

—

## 7) Configurație (config/preprocessing_config.yaml)
Chei importante:
- `dataset.subset`: ex. `combined_reduced` | `combined` | `email` | `sms` | `url` | `html`
- `preprocessing`: `lowercase`, `strip_html`, `remove_urls`, `remove_duplicates`, `normalize_whitespace`
- `missing_values.drop_empty_text`: eliminare rânduri cu text gol
- `outliers.length_clip.enabled` + `max_chars`: tăiere texte foarte lungi
- `split`: rapoarte, `stratify`, `random_seed`, `shuffle`
- `save_paths`: directoarele pentru output
- `logging.save_reports` + `reports_dir`: rapoarte Markdown

—

## 8) Structura proiectului
```
proiect-rn/
├── README.md                      # acest document (unic)
├── config/
│   └── preprocessing_config.yaml
├── data/
│   ├── raw/                       # phishing_{subset}.csv (input CSV standardizat)
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── docs/
│   └── datasets/
│       ├── EDA_report_{subset}.md
│       └── plots/
├── src/
│   ├── data_acquisition/
│   │   └── download_hf_dataset.py
│   ├── preprocessing/
│   │   ├── eda.py
│   │   └── preprocess_and_split.py
│   └── neural_network/            # rezervat pentru etapa următoare
└── requirements.txt
```

—

## 9) Fișiere rezultate (după rulare)
- EDA: `docs/datasets/EDA_report_combined_reduced.md`, grafice în `docs/datasets/plots/`
- Preprocesare: `data/processed/processed_combined_reduced.csv`
- Split: `data/train/train_combined_reduced.csv`, `data/validation/validation_combined_reduced.csv`, `data/test/test_combined_reduced.csv`
- Raport preprocesare: `docs/datasets/Preprocess_report_combined_reduced.md`

—

## 10) Troubleshooting (erori frecvente)
- datasets v3 „Dataset scripts are no longer supported…”: folosește versiunea curentă de script (are `trust_remote_code=True`) sau setează `HF_DATASETS_TRUST_REMOTE_CODE=1`.
- Subset invalid: listează config-urile disponibile și alege unul din listă.
- `FileNotFoundError` la EDA sau Preprocess: verifică existența `data/raw/phishing_{subset}.csv` sau folosește `--input` cu calea corectă; conversia JSON→CSV este în Pasul 0.
- `ModuleNotFoundError`: reinstalează dependențele asigurându-te că `pip` corespunde cu `python` (ex. `python3 -m pip install -r requirements.txt`).
- Probleme la salvarea graficelor: asigură-te că există `docs/datasets/plots/` (scriptul îl creează automat) și ai permisiuni de scriere.

—

## 11) Stare Etapă
- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată complet (după rularea Preprocess & Split)

—

## 12) Ce urmează (Etapa următoare)
- Antrenarea unui model de clasificare text (ex. DistilBERT) folosind `data/train/`, evaluare pe `validation/test`, logare metrici (accuracy, precision, recall, F1) și salvare model.

Sugerezi modificări sau vrei să rulăm împreună pașii? Urmează exact secțiunea „Rulare rapidă (3 pași)”.