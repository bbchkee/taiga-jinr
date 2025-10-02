
# QUICKSTART

1. Install requeirements: `pip install -r requirements.txt`.
2. Run pipeline:
   ```bash
   python scripts/run_pipeline.py --model-path /path/to/model_csv --exp-path /path/to/exp_csv
   ```
3. Script will plot theta^2 excess and energy spectrum to `plots/`

## Input data
CSV contains tables (основные):
- Model: `size`, `dist[0]`, `width[0]`, `length[0]`, `energy`, `numb_pix`
- Experiment: `size`, `dist1`, `width`, `length`, `alpha1`, `alpha2`