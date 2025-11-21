# Descrierea setului de date (Etapa 3)

Descriere completată pentru dataset-ul ales: Hugging Face – ealvaradob/phishing-dataset.

## 1. Sursa datelor
- Origine: public – Hugging Face (multi-sursă: e-mail, SMS, URL, HTML)
- Modul de achiziție: ☑️ Fișier extern (prin librăria `datasets`) / ☑️ Generare programatică (download API)
- Perioada/condițiile colectării: dataset public agregat (perioade multiple, diverse surse)
- Linkuri sursă:
  - https://huggingface.co/datasets/ealvaradob/phishing-dataset

## 2. Caracteristicile dataset-ului
- Număr total de observații: variază în funcție de subset (ex.: „combined”, „combined_reduced”). Se va raporta după descărcare.
- Număr de caracteristici (features): 2 principale (text + etichetă); pot exista câmpuri suplimentare în anumite subseturi.
- Tipuri de date: ☑️ Text (mesaje e-mail/SMS/URL/HTML)
- Format fișiere: ☑️ Arrow/Parquet intern (prin `datasets`); export opțional CSV/JSON pentru procesare ulterioară

## 3. Descrierea fiecărei caracteristici (generic pentru subsetele text)
| Caracteristică | Tip   | Unitate | Descriere                                   | Domeniu valori            |
|----------------|-------|---------|---------------------------------------------|---------------------------|
| text           | text  | –       | Conținutul mesajului (email/SMS/URL/HTML)   | șir de caractere          |
| label          | categ | –       | Eticheta clasei                             | {phishing, legit/ham}     |

Notă: denumirile exacte ale câmpurilor pot varia pe subset; se vor confirma după încărcare (ex.: `label`, `target`, `class`).

## 4. Note privind calitatea datelor (EDA – de completat după încărcare)
- Distribuția claselor (phishing vs. legitim): [de raportat]
- Valori lipsă în câmpul `text`: [procente]
- Lungimea textelor (medie, mediană, IQR): [valori]
- Duplicări (texte identice): [procent]
- Outlieri (lungimi extreme): [metodă IQR/percentile]

## 5. Structura folderelor (Etapa 3)
- data/raw/ – dump inițial (ex.: export CSV/JSON al subsetului selectat)
- data/processed/ – rezultate după curățare/transformări (HTML→text, lowercasing opțional, etc.)
- data/train/, data/validation/, data/test/ – împărțirea finală pentru model

## 6. Reproduceri și configurări
- Parametrii preprocesare: vezi `config/preprocessing_config.yaml`
- Versiuni librării: vezi `requirements.txt`

## 7. Subset vizat inițial
- Propunere: `combined_reduced` (balansat) pentru EDA rapidă; ulterior se poate trece la subset complet.


<!-- EDA_SUMMARY_START -->
### Rezumat EDA (subset: combined_reduced)
- Rânduri: 77677 | Coloane: 2
- Duplicate după text: 109
- Distribuție etichete:
  - 0: 44975 (0.579)
  - 1: 32702 (0.421)
<!-- EDA_SUMMARY_END -->
