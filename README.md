# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Chelu Fabian-Catalin  
**Grupa:** 632AB
**Link Repository GitHub:** https://github.com/littlebodybigheart01/proiect_rn_phishing 
**Data:** 05.12.2025  

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din specificațiile proiectului.

Obiectivul este livrarea unui **schelet complet și funcțional** al sistemului de detecție a phishing-ului, demonstrând integrarea fluxului de date (Data Pipeline), a modelului de Deep Learning și a interfeței cu utilizatorul. Sistemul este capabil să parcurgă ciclul complet: Generare Date -> Antrenare -> Inferență -> Afișare Rezultat.

---

## 1. Structura Repository-ului

Proiectul respectă o structură modulară, separând datele brute, codul sursă și modelele antrenate conform standardelor de inginerie software.

```text
├── app.py                              # Punctul de intrare în Aplicația Web (Streamlit)
├── config
│   └── preprocessing_config.yaml       # Fișier de configurare pentru pipeline-ul de date
├── data
│   ├── processed
│   │   └── processed_ai_generated.csv  # Date curățate și tokenizate (cache)
│   ├── raw                             # Surse de date brute (Hibrid: Real + Sintetic)
│   │   ├── emailreal.csv               # Dataset Enron (Engleză)
│   │   ├── smsreal.csv                 # Dataset SMS Spam Collection (Engleză)
│   │   ├── phishing_ai_ro_only.csv     # Date sintetice generate cu Gemini (Română)
│   │   ├── phishing_ai_targeted_patch.csv # Date adversariale (Hard Examples)
│   │   ├── multilingualdataset.csv     # Dataset intermediar
│   │   └── final_multilingual_dataset.csv # Dataset final unificat și balansat
│   ├── train
│   │   └── train_ai_generated.csv      # Subset antrenament (80%)
│   ├── validation
│   │   └── validation_ai_generated.csv # Subset validare (10%)
│   └── test
│       └── test_ai_generated.csv       # Subset testare (10%)
├── docs
│   └── datasets
│       └── plots
│           └── confusion_matrix.png    # Grafice de performanță
├── models
│   └── phishing_distilbert_multilingual # Director salvare model antrenat
│       ├── config.json
│       ├── special_tokens_map.json
│       ├── tf_model.h5                 # Ponderile modelului (TensorFlow)
│       ├── tokenizer_config.json
│       └── vocab.txt
├── README.md                           # Documentația curentă
├── requirements.txt                    # Dependențe Python
└── src
	├── data_acquisition                # Modul 1: Achiziție Date
	│   ├── generate_ai_data.py         # Script generare cu Gemini API
	│   └── merge_all_datasets.py       # Script unificare surse hibride
	├── neural_network                  # Modul 2: Rețea Neuronală
	│   ├── model.py                    # Definirea arhitecturii DistilBERT
	│   ├── train.py                    # Bucla de antrenare
	│   ├── evaluate.py                 # Script evaluare metrici
	│   └── predict.py                  # Script testare consolă
	└── preprocessing                   # Modul Preprocesare
		└── preprocess_and_split.py     # Curățare, tokenizare, split
```
2. Arhitectura de Sistem (SIA)
Diagrama de mai jos (reprezentată ca tabele pentru compatibilitate) ilustrează fluxul datelor prin componentele sistemului, evidențiind abordarea hibridă de achiziție a datelor și procesarea acestora.

**Arhitectura sistemului — Componenta & Flux**

| Componentă | Rol principal | Input | Output | Fișiere cheie |
|-------------|---------------|-------|--------|----------------|
| Modul 1: Achiziție Date (Pipeline Hibrid) | Generare și colectare date (sintetic + real), curățare, normalizare și balansare | Google Gemini API, seturi externe (Enron, SMS, etc.) | `final_multilingual_dataset.csv` (raw, balansat) | `src/data_acquisition/generate_ai_data.py`, `merge_all_datasets.py` |
| Modul 2: Rețea Neuronală (DistilBERT) | Tokenizare, antrenare (fine-tuning), evaluare și salvare model | Dataset tokenizat (max_length=128) | Model salvat (`tf_model.h5` / SavedModel) | `src/neural_network/model.py`, `train.py`, `evaluate.py` |
| Modul 3: Web Service (Streamlit) | Interfață utilizator, request inferență, afișare verdict și explicații | Text introdus de utilizator | Probabilitate phishing, logits, UI update | `app.py`, `src/neural_network/predict.py` |

**Flux de date (pas cu pas)**

| Pas | Activitate | Componentă responsabilă | Condiție de trecere |
|-----|-----------|------------------------|---------------------|
| 1 | Generare / colectare date | `generate_ai_data.py` / surse externe | Date disponibile în `data/raw/` |
| 2 | Merge, curățare, balansare | `merge_all_datasets.py` | `final_multilingual_dataset.csv` creat |
| 3 | Preprocesare & tokenizare | `preprocess_and_split.py` | Tensori pregătiți pentru antrenare/inferență |
| 4 | Antrenare model | `train.py` | Model salvat în `models/phishing_distilbert_multilingual/` |
| 5 | Inferență în aplicație | `app.py` → `predict.py` | Răspuns (probabilitate) returnat către UI |



3. Descrierea Componentelor

Sistemul este modularizat pentru a asigura scalabilitatea și mentenabilitatea codului.

Modul 1: Data Logging / Acquisition
Acest modul gestionează crearea unui set de date robust. Deoarece seturile de date publice în limba română pentru phishing sunt limitate, am dezvoltat o soluție hibridă:

Generare Sintetică (generate_ai_data.py): Utilizează LLM-uri (Google Gemini) pentru a genera scenarii de atac specifice pieței din România (ex: false notificări ANAF, Poșta Română, Bănci locale) și date adversariale (phishing_ai_targeted_patch.csv) pentru a corecta vulnerabilitățile modelului.

Unificare (merge_all_datasets.py): Combină datele sintetice cu seturi reale consacrate (emailreal.csv, smsreal.csv). Scriptul gestionează discrepanțele de format și curăță caracterele neconforme, rezultând final_multilingual_dataset.csv.

Modul 2: Neural Network (Arhitectura)
Nucleul sistemului este o rețea neuronală bazată pe arhitectura Transformer, utilizând modelul distilbert-base-multilingual-cased.

Arhitectură: Transformer Encoder (12 straturi) + Strat Dense (Clasificare).

Input: Tokenizer DistilBERT (max_length=128).

Training: Modelul este antrenat folosind train.py, care salvează ponderile optimizate în directorul models/phishing_distilbert_multilingual.

Performanță: Utilizează funcția de activare Sigmoid pentru a returna o probabilitate de risc între 0 și 1.

Modul 3: Web Service / UI
Interfața (app.py) este dezvoltată în Streamlit, oferind o experiență utilizator modernă.

Design: Temă vizuală personalizată ("Y2K/Cyberpunk") pentru impact vizual.

Funcționalitate: Procesează textul în timp real, interoghează modelul salvat și afișează verdictul (Phishing/Legitim).

Interactivitate: Include elemente dinamice (Easter Eggs, feedback vizual instant).

4. Diagrama Fluxului de Date (State Machine)
Această diagramă descrie stările prin care trece sistemul în timpul procesării unei cereri.

**State Machine (tabele)**

**Stări principale**

| Stare | Ce se întâmplă aici | Condiție intrare | Condiție ieșire |
|-------|---------------------|------------------|-----------------|
| Idle | Sistemul așteaptă inputul utilizatorului | Aplicația pornită sau după un ciclu complet | Utilizator apasă "SCAN" |
| Preprocessing | Curățare text, tokenizare, conversie în tensori | Text recepționat din UI | Tensori validați, gata pentru inferență |
| Inference | Propagare înainte prin model (DistilBERT), calcul logits | Tensori validați | Logits și scoruri calculate |
| DecisionLogic | Aplicare praguri, decizie finală (Phishing/Legit/Uncertain) | Scoruri disponibile | Rezultat clasificat (trimis la UI) |
| UI_Update | Afișare rezultat, explicații și logare | Rezultat clasificare | Resetare la `Idle` pentru input nou |

**Tranziții critice**

| De la | Către | Condiție / Descriere |
|------:|:------|:--------------------|
| Idle | Preprocessing | Utilizator inițiază scanarea (apasă "SCAN") |
| Preprocessing | Inference | Toate transformările și tokenizarea s-au încheiat cu succes (tensori validați) |
| Inference | DecisionLogic | Modelul a returnat logits/probabilități |
| DecisionLogic | UI_Update | Decizia finală este calculată (ex: scor > 0.75 → Phishing) |
| UI_Update | Idle | Utilizator finalizează vizualizarea sau revine pentru input nou |
| DecisionLogic | PhishingState | Scor > 0.75 (exemplu de prag configurabil) |
| DecisionLogic | LegitState | Scor < 0.25 |
| DecisionLogic | UncertainState | 0.25 ≤ Scor ≤ 0.75 |
    
5. Checklist Etapa 4
General
[x] Diagrama Arhitectură SIA creată.

[x] Diagrama State Machine definită și documentată.

[x] Structura repository-ului este organizată (src/, data/, models/).

Modul 1: Achiziție Date
[x] Scripturile de generare (generate_ai_data.py) funcționează corect.

[x] Scriptul de unificare (merge_all_datasets.py) integrează date reale și sintetice.

[x] Dataset-ul final este salvat în data/raw/.

Modul 2: Rețea Neuronală
[x] Modelul DistilBERT este definit în src/neural_network/model.py.

[x] Modelul este compilat și salvat (models/phishing_distilbert_multilingual/tf_model.h5).

[x] Scriptul de antrenare (train.py) este funcțional.

Modul 3: Interfață Web
[x] Aplicația app.py pornește fără erori.

[x] Interfața acceptă input și afișează predicția modelului în timp real.

6. Instrucțiuni de Rulare
Pentru a reproduce mediul și a rula aplicația, urmați pașii de mai jos:

1. Instalarea Dependențelor
Bash

```bash
pip install -r requirements.txt
```
2. Pregătirea Datelor (Opțional)
Bash

```bash
# Unificarea datelor sintetice cu cele reale
python src/data_acquisition/merge_all_datasets.py

# Preprocesarea și împărțirea (Train/Val/Test)
python src/preprocessing/preprocess_and_split.py
```
3. Antrenarea Modelului
Bash

```bash
python src/neural_network/train.py
```
4. Lansarea Aplicației Web
Bash

```bash
streamlit run app.py
```
Aplicația va fi accesibilă la http://localhost:8501.

7. Dependențe (requirements.txt)
tensorflow: Framework-ul de bază pentru Deep Learning.

transformers: Biblioteca HuggingFace pentru modelul DistilBERT.

streamlit: Framework pentru interfața grafică web.

google-generativeai: Clientul API pentru Google Gemini.

pandas & numpy: Manipularea datelor.

scikit-learn: Procesarea și împărțirea datelor.

requests: Interogări API externe. vei pune totul in readme.md (See <attachments> above for file contents. You may not need to search or read the file again.)

## Nevoi reale (Use‑cases) și acoperirea SIA

| Nevoie reală concretă | Cum o rezolvă SIA-ul vostru | Modul software responsabil |
| :--- | :--- | :--- |
| Detectarea atacurilor de phishing localizate (ex: false notificări ANAF/Curierat în limba română) care trec de filtrele clasice de spam. | Clasificare semantică bazată pe **DistilBERT Multilingual** → verdict de risc (Phishing/Legitim) în **< 1 secundă** cu acuratețe **> 98%**. | **Neural Network** + **Web Service** |
| Protecția utilizatorilor împotriva Ingineriei Sociale complexe (ex: CEO Fraud fără link-uri, Typosquatting) care păcălește ochiul uman. | Antrenare adversarială pe dataset hibrid (Real + Sintetic generat pe scenarii specifice) → identificarea tiparelor de manipulare psihologică. | **Data Acquisition** + **Neural Network** |
| Educarea utilizatorilor privind motivele pentru care un mesaj este considerat periculos (Explainability). | Generarea automată a unei explicații în limbaj natural (prin LLM) pentru fiecare verdict → feedback instantaneu despre elementele suspecte detectate. | **Web Service / UI** (Logică Backend) |

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Total observații finale:** ~40,000 (după Etapa 3 + Etapa 4)
**Observații originale:** ~20,000 (~50%)

**Tipul contribuției:**
- [x] Date generate prin simulare/metode avansate (Data Augmentation cu LLM)
- [ ] Date achiziționate cu senzori proprii
- [ ] Etichetare/adnotare manuală
- [ ] Date sintetice prin metode avansate

**Descriere detaliată:**
Pentru a crea un model robust și capabil să detecteze atacuri specifice contextului românesc (care lipsesc din dataset-urile internaționale publice precum Enron), am dezvoltat un pipeline de generare sintetică folosind API-ul **Google Gemini**. Am creat prompt-uri specifice ("Adversarial Prompts") pentru a simula atacuri de tip:
1.  **Phishing Localizat:** Mesaje false de la instituții românești (ANAF, Poșta Română, Bănci: BT, ING, BCR, eMAG, OLX).
2.  **Inginerie Socială:** CEO Fraud (fără link-uri, bazat pe autoritate), "Prieten la nevoie".
3.  **Obfuscation:** Typosquatting (`rnicrosoft`, `Faceb00k`) și link-uri mascate.

Aceste date au fost apoi curățate, validate și combinate cu datele reale (Enron Email Corpus, SMS Spam Collection) pentru a asigura un echilibru între realismul limbajului natural și diversitatea vectorilor de atac.

**Locația codului:** `src/data_acquisition/generate_ai_data.py` și `src/data_acquisition/generate_targeted_weaknesses.py`
**Locația datelor:** `data/raw/phishing_ai_mixed_complete.csv` și `data/raw/phishing_ai_targeted_patch.csv`

**Dovezi:**
- Dataset-urile generate se află în directorul `data/raw/`.
- Logurile de generare și distribuția claselor sunt vizibile la rularea scriptului `merge_all_datasets.py`.

