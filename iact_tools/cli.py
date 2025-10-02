
import argparse, os
from .data_loading import GammaDataLoader
from .reconstruction import GammaSpectrumReconstructor
from .analysis import SpectrumAnalyzer

def main():
    ap = argparse.ArgumentParser(description="IACT pipeline: load -> fit -> reconstruct -> plots")
    ap.add_argument("--model-path", required=True, help="Path to model CSV folder")
    ap.add_argument("--exp-path", required=True, help="Path to experiment CSV folder")
    ap.add_argument("--exp-pattern", default="*hillas_14_7.0fix.csv", help="Glob for experiment files")
    args = ap.parse_args()

    loader = GammaDataLoader(args.model_path, args.exp_path)
    model_df = loader.load_model_data()
    exp_df = loader.load_experiment_data(args.exp_pattern)

    recon = GammaSpectrumReconstructor(model_df)
    recon.fit_energy_size(out_dir='plots')
    exp_df = recon.reconstruct_energy(exp_df)
    exp_df = recon.filter_experiment_data(exp_df)

    analyzer = SpectrumAnalyzer(exp_df)
    analyzer.plot_spectrum()
    analyzer.plot_theta2_distribution()

    out_csv = os.path.join("outputs", "exp_with_energy.csv")
    os.makedirs("outputs", exist_ok=True)
    exp_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
