
#!/usr/bin/env python3
import argparse, os
import pandas as pd
from iact_tools.data_loading import GammaDataLoader
from iact_tools.reconstruction import GammaSpectrumReconstructor as GSR
from iact_tools.analysis import SpectrumAnalyzer

def main():
    ap = argparse.ArgumentParser(description="IACT pipeline: load -> fit -> reconstruct -> plots")
    ap.add_argument("--model-path", required=True, help="Path to model CSV folder")
    ap.add_argument("--exp-path", required=True, help="Path to experiment CSV folder")
    ap.add_argument("--exp-pattern", default="*hillas_14_7.0fix.csv", help="Glob for experiment files")
    ap.add_argument("--dump", default=False, action = "store_true", help = "Dump tables with calculated parameters")

    ap.add_argument("--sums", default=None, help="Path to sums file (energies of ALL thrown MC events). If provided, compute A_eff.")
    ap.add_argument("--throw-radius", type=float, default=500.0, help="Throw radius in meters for A_eff.")
    ap.add_argument("--eff-bins", type=int, default=20, help="Number of log-energy bins for A_eff.")
    ap.add_argument("--unfold", action="store_true", help="Do Tikhonov unfolding")
    ap.add_argument("--tau", type=float, default=1.0, help="Tikhonov regularization strength (tau)")
    ap.add_argument("--tau-auto", action="store_true", help="Auto-pick tau by L-curve (overrides --tau)")


    args = ap.parse_args()

    loader = GammaDataLoader(args.model_path, args.exp_path) # init loader
    model_df = loader.load_model_data() # load model data (take path with .csv fits)
    exp_df = loader.load_experiment_data(args.exp_pattern) # load experiment data (take path with .csv fits, you can use regexp for filenames)

    recon = GSR(model_df) # initializing
    recon.fit_energy_size(out_dir='plots') # calculate dependence size vs energy (linear fit) by model data
    
    exp_df = recon.apply_good_cut(exp_df) # quality cuts (weather, edge, pointing ...)
    exp_df = recon.add_background_alphas_dists_thetas(exp_df) # calculate Hillas geometric pars for background points
    exp_df = recon.classify_by_cuts(exp_df) # add gamma_like flags for source and backgound points. For this is needed to add background alphas dists thetas
    
    exp_df = recon.reconstruct_energy(exp_df) # reconstruct energy

    # --- Optional: compute effective area if sums provided ---
    if args.sums is not None:
        try:
            from iact_tools.effective_area import compute_effective_area_from_files
            from iact_tools.effective_area import write_effective_area_products
            from iact_tools.reconstruction import filter_model_data
            # Use model energies (PASSED) for A_eff
            eff_df = compute_effective_area_from_files(
                args.sums,
                model_df,
                n_bins=args.eff_bins,
                radius_m=args.throw_radius,
                energy_col="energy",
                energy_range=None,
                model_filter=filter_model_data,
            )
            os.makedirs("outputs", exist_ok=True)
            csv_path = write_effective_area_products(eff_df, out_dir="outputs", basename="effective_area")
            print(f"[A_eff] Saved table/plot to: {csv_path}")
        except Exception as e:
            print(f"[A_eff] Failed to compute effective area: {e}")

    # now we have our df filled with all additional values
    analyzer = SpectrumAnalyzer(exp_df) # init analyzer

    # --- Optional: Unfolding ---
    if args.unfold:
        try:
            from iact_tools.reconstruction import GammaSpectrumReconstructor
            from iact_tools.unfolding import run_unfolding_pipeline
            # Ensure reconstructed energies exist for both exp and model
            model_df_reco = recon.reconstruct_energy_for_model_data(model_df)
            # Load A_eff table produced earlier (or compute if absent and sums provided)
            eff_csv_path = os.path.join("outputs", "effective_area.csv")
            if not os.path.exists(eff_csv_path) and args.sums is not None:
                from iact_tools.effective_area import compute_effective_area_from_files, write_effective_area_products
                eff_df_tmp = compute_effective_area_from_files(args.sums, model_df_reco, n_bins=args.eff_bins, radius_m=args.throw_radius, energy_col="energy")
                write_effective_area_products(eff_df_tmp, out_dir="outputs", basename="effective_area")
            eff_df = pd.read_csv(eff_csv_path)
            unfolded_flux_df, R, M = run_unfolding_pipeline(exp_df, model_df_reco, eff_df, tau=(None if args.tau_auto else args.tau))
            os.makedirs("outputs", exist_ok=True)
            unfolded_csv = os.path.join("outputs", "unfolded_spectrum.csv")
            unfolded_flux_df.to_csv(unfolded_csv, index=False)
            print(f"[Unfolding] Saved: {unfolded_csv}")
        except Exception as e:
            print(f"[Unfolding] Failed: {e}")

    # --- Optional: build physical spectrum if effective_area products exist ---
    eff_csv_path = os.path.join("outputs", "effective_area.csv")
    if os.path.exists(eff_csv_path):
        try:
            #import pandas as pd
            from iact_tools.analysis import compute_physical_spectrum, plot_physical_spectrum, compute_observation_time_minutes
            eff_df = pd.read_csv(eff_csv_path)
            spec_df = compute_physical_spectrum(exp_df, eff_df, energy_col_reco="reconstructed_energy", theta2_col="theta2_1", theta2_max=0.05)
            os.makedirs("outputs", exist_ok=True)
            spec_csv = os.path.join("outputs", "physical_spectrum.csv")
            spec_df.to_csv(spec_csv, index=False)
            plot_physical_spectrum(spec_df, out_png="plots/physical_spectrum.png", logy=True)
            T_min = compute_observation_time_minutes(exp_df, "por", 2.0)
            print(f"[Spectrum] Built physical spectrum. Observation time ≈ {T_min:.1f} min. Saved: {spec_csv}")
        except Exception as e:
            print(f"[Spectrum] Failed to build physical spectrum: {e}")

    analyzer.plot_spectrum() # plot energy spectrum (at this moment without unfolding and effective area TODO)
    analyzer.plot_theta2_distribution() # plots theta2 excess (source - mean(background))

    if args.dump: # save this huge file do disk if you want
        out_csv = os.path.join("outputs", "exp_with_energy.csv")
        os.makedirs("outputs", exist_ok=True) 
        exp_df.to_csv(out_csv, index=False) 
        print(f"Saved: {out_csv}") 

if __name__ == "__main__":
    main()
