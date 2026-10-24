# MGC Developments — AI Developer & Engineer Assessment

A small, explainable implementation of all four assessment parts. The assessment brief explicitly prioritizes **running code, grounded answers, conflict handling, sensible SQL, honest ML decisions, and a working minimal web page** over styling or deployment.

## Project structure

```text
mgc-ai-assessment/
├── data/
│   ├── 01_brochure.md
│   ├── 02_price_list_payment_plan.md
│   ├── 03_booking_policy_faq.md
│   └── leads.csv
├── part1_rag/
│   ├── rag_engine.py
│   └── cli.py
├── part2_database/
│   ├── schema.sql
│   └── queries.sql
├── part3_ml/
│   └── train.py
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── tests/
└── README.md
```

## Quick start

From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python part3_ml/train.py
python part2_database/load_sqlite.py
python app.py
```

Open `http://127.0.0.1:5000` for the web UI.

## Run from scratch

### 1. Create environment

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Part 3 first (creates the model used by the web app)

```bash
python part3_ml/train.py
```

The script prints the positive-class rate and Average Precision, then saves:
`part3_ml/lead_scoring_model.joblib`

### 3. Run Part 1 from CLI

```bash
python -m part1_rag.cli "What's the base price of a 2-bed in Block B?"
python -m part1_rag.cli "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?"
python -m part1_rag.cli "What's the transfer fee?"
python -m part1_rag.cli "What's the rental yield on a 1-bed?"
python -m part1_rag.cli "Who is the anchor tenant?"
```

### 4. Run Part 4

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

The page supports both:
- document questions with source display
- lead scoring with the baseline model

## Part 1 — design and hard cases

This uses a lightweight local TF-IDF retriever because it is fast, deterministic, explainable, and requires no paid API or API key.

The important behaviour is not just retrieval:
1. **Grounding:** answers come from the supplied MGC documents.
2. **Stacked calculation:** the floor, corner, and Margalla premiums are cumulative.
3. **Conflict:** transfer fee is deliberately reported as a conflict rather than silently selecting one document.
4. **Missing information:** rental yield is not invented; the FAQ says to refer the customer to the marketing manager.
5. **Unconfirmed information:** the brochure says anchor-tenancy discussions are ongoing and no tenant is confirmed.

Expected hard-case outputs:
- 2-bed Block B base price: **PKR 22,425,000**
- Floor 15 + corner + Margalla-facing: **PKR 25,340,250** (= 22,425,000 × 1.13)
- Transfer fee: **2% in Price List vs 2.5% in Booking Policy; confirm with marketing manager**
- Rental yield: **not published; do not invent**
- Anchor tenant: **not confirmed**

## Part 2 — data observations and schema

The supplied CSV has **9,160 rows and 20 columns**.

Important observations:
- `lead_id` is unique in the supplied file.
- `crm_record_hash` has **160 duplicated hash values (320 rows)**, which is the strongest duplicate identity signal in the dump.
- Several categorical values have inconsistent spelling/case, especially city values such as `Islamabad`, `ISLAMABAD`, `ISB`; the ML script normalizes these.
- Missing values exist in `area`, `budget_pkr_lac`, `bedrooms`, `first_response_minutes`, and `agent_experience_years`.

The clean schema uses:
- `lead_id` as primary key
- `crm_record_hash` as a unique identity constraint to prevent duplicate CRM records
- appropriate numeric/boolean/text/timestamp types
- nullable columns where the source data is missing

The loader first keeps the raw CSV in `crm_import_raw` for auditing, then deduplicates by `crm_record_hash` before inserting into the constrained `leads` table. This makes the schema constraint real rather than only documented.

The duplicate query runs against `crm_import_raw`, while the conversion-rate query runs against the clean `leads` table.

For a quick local demonstration:
```bash
python part2_database/load_sqlite.py
```
This creates `part2_database/mgc.db`, applies `schema.sql`, and executes the two core analyses. The generated database is ignored by Git because it can be recreated from the CSV.

## Part 3 — ML decisions

### What I cleaned/fixed
- Normalize obvious city spelling/case variants.
- Keep missing values and impute them in the preprocessing pipeline rather than dropping many rows.
- One-hot encode categorical fields.
- Standardize numeric fields.

### What I deliberately did not use
I excluded:
- `lead_id` — identifier, not a predictive feature.
- `crm_record_hash` — duplicate identity key, not a feature.
- `agent_experience_years` — agent attribute rather than lead intent; using it could make the score depend on who owns the lead.
- `first_response_minutes`, `calls_made`, `total_call_seconds`, `whatsapp_replies`, `site_visits` — these are post-contact activity variables. The business problem is deciding which leads to call first, so using future sales-team activity would create leakage at scoring time.
- `token_amount_received_pkr` — strongly downstream of the conversion process and therefore a leakage risk.

I kept basic lead/profile information plus `has_financing_approved`, because it can plausibly be known before prioritization.

### Baseline model
- Logistic Regression
- no hyperparameter tuning
- `class_weight="balanced"` because conversion is highly imbalanced
- stratified 80/20 train/test split
- metric: **Average Precision (PR-AUC)**

The supplied data has **634 converted out of 9,160 leads (~6.92%)**. With this imbalance, accuracy could look high while being operationally unhelpful. Average Precision focuses on precision/recall quality for the positive class and is appropriate for a lead-ranking problem.

Run:
```bash
python part3_ml/train.py
```

The script prints the exact held-out Average Precision from the current dataset. I intentionally do not hard-code a metric value in this README.

## Part 4 — web

A single Flask page ties both useful pieces together. It is intentionally bare because the assessment says styling is not evaluated.

## What I would do next with more time

- Replace the local TF-IDF answer layer with a small LLM generator constrained by retrieved passages, while keeping the same refusal/conflict rules.
- Add a proper ingestion step that loads the original PDFs directly and preserves page numbers for citations.
- Add a clean/raw database staging table and an explicit deduplication process.
- Evaluate lead scoring with time-based validation if production data spans time.
- Add threshold/Top-K evaluation tied to how many leads the sales team can actually call.

## Assessment compliance

The brief says AI tools are allowed, but every submitted line must be explainable. This project therefore favors small modules, deterministic rules for the high-risk cases, and a simple baseline model over unnecessary infrastructure.

## Tests

Install the development requirements and run the automated checks:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover document retrieval edge cases, safe database replacement, the leakage-conscious training pipeline, and form validation/error responses. A GitHub Actions workflow runs this suite on pushes and pull requests.

