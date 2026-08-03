# IPL Match Predictor

An IPL pre-match prediction project. The first two phases establish a reproducible
data pipeline and a team-only match-context resolver.

## Data sources

- IPL ball-by-ball data: [Cricsheet IPL JSON](https://cricsheet.org/downloads/ipl_json.zip)
- Men's T20 international ball-by-ball data: [Cricsheet T20 JSON](https://cricsheet.org/downloads/t20s_json.zip)

Cricsheet is downloaded on demand; large raw and processed datasets are deliberately
not committed to the repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.ingestion.download_cricsheet --dataset ipl
python -m src.ingestion.parse_cricsheet --dataset ipl
```

Optional international T20 data:

```bash
python -m src.ingestion.download_cricsheet --dataset t20i
python -m src.ingestion.parse_cricsheet --dataset t20i
```

## Team-only match context

Place a verified IPL schedule in `data/external/ipl_schedule.csv` with:

```text
date,team_a,team_b,venue
2026-04-01,Chennai Super Kings,Mumbai Indians,MA Chidambaram Stadium
```

Then resolve the matchup:

```bash
python -m src.context.match_context "CSK" "MI"
```

The resolver uses the next matching fixture on or after today. If no schedule record
exists, it returns a clearly labelled neutral-venue, hypothetical context; it never
invents a venue or date.

## Build features and train models

```bash
python -m src.features.build_features
python -m src.models.train
python -m src.models.predict CSK MI
```

The feature builder uses only matches before each target match. Training uses 2025
as validation and 2026 as a final unseen test season. The saved ensemble combines
Logistic Regression, Random Forest, and calibrated XGBoost probabilities.

## Explainability, player intelligence, and grounded previews

```bash
python -m src.insights CSK MI
```

This returns local SHAP factors, an inferred (not confirmed) likely squad based on
recent IPL appearances, player impact scores, and a preview grounded in retrieved
player records.

## React dashboard

The React dashboard is in `frontend/`. Start the API from the project root:

```bash
python -m uvicorn src.api.main:app --reload
```

In a second terminal, start the React client:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). No deployment is configured. See [the model card](docs/MODEL_CARD.md) for limitations and evaluation guidance.

## Testing

```bash
python -m pytest -q
```

The tests cover Cricsheet parsing, team-only fixture fallback, leakage-safe feature construction, retrieval ranking, and grounded-preview evidence.
