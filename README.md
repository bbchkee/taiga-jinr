# iact-tools

A set of utilities for analyzing TAIGA-IACT (Cherenkov) data.
Includes data loading, energy reconstruction, basic cuts, spectrum building, and theta^2 distributions.

## Structure

- `iact_tools/` - package code
  -  `data_loading.py` - CSV loading and quick histograms
  -  `models.py` - mathematical models and theta^2
  -  `reconstruction.py` - energy reconstruction and cuts
  -  `analysis.py` - spectrum and theta^2 plots
  -  `regressor.py` - Torch-based energy regressor
  -  `utils.py` - helpers
  -  `scripts/run_pipeline.py` - CLI script
  -  `docs/` - documentation
  -  `tests/` - tests

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --model-path /path/to/model_csv --exp-path /path/to/exp_csv
# или через entrypoint после установки:
# pip install .
# iact-run --model-path ... --exp-path ...
```

Output plots are stored in plots/, tables in outputs/.

## License
GNU GPL v3

