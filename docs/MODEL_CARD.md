# Model card

## Intended use

This model estimates IPL pre-match win probabilities for two selected teams. It is a portfolio analytics project, not betting, financial, or selection advice.

## Data and splits

The project derives records from Cricsheet IPL ball-by-ball JSON. Historical features are calculated only from matches before the match being predicted. It trains through 2024, validates on 2025, and reports a final test on 2026.

## Inputs and limitations

Inputs are two teams. If a verified schedule entry is unavailable, the application uses a neutral venue. Player lists are inferred from recent IPL appearances and must not be treated as confirmed playing XIs. The model does not yet include toss, weather, injuries, or final lineups.

## Metrics

Use log loss and Brier score alongside accuracy because the product exposes probabilities. The exact latest metrics are written to `models/evaluation.json` after training.
