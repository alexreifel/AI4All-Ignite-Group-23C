# Analysis notebooks

The original single notebook has been split into six staged notebooks, one per pipeline section, so each part of the analysis lives somewhere clear instead of one long file. Run them in order:

1. `01_loading_data.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_eda.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_model_building.ipynb`
6. `06_evaluation.ipynb`

Open each with Jupyter from the repo root (`jupyter notebook notebooks/01_loading_data.ipynb`) and run all cells top to bottom before moving to the next one. Each notebook loads the objects it needs (dataframes, fitted models, feature lists) from the notebook before it, and saves what the next one needs, via small joblib files under `notebooks/_state/`. That folder is generated automatically on first run and is gitignored, not committed.

`01_loading_data.ipynb` reads `data/american_bankruptcy.csv`, so `data/` must sit next to `notebooks/` at the repo root (this is already the case if you cloned the repo normally).

This split keeps every line of the original analysis, in the original order, just broken at its existing section boundaries (the same six sections documented in the project's `CLAUDE.md`).
