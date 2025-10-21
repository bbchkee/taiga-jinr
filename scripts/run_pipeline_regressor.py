#!/usr/bin/env python3
import argparse, os
import pandas as pd

from iact_tools.data_loading import GammaDataLoader
from iact_tools.reconstruction import fixed_energy_bins, GammaSpectrumReconstructor
from iact_tools.analysis import SpectrumAnalyzer
from iact_tools.regressor import GammaNNRegressor

def main():
    ap = argparse.ArgumentParser(description="IACT pipeline (NN regressor)")
    ap.add_argument("--model-path", required=True, help="Directory with MODEL CSV(s)")
    ap.add_argument("--exp-path",   required=True, help="Directory with EXP CSV(s)")
    ap.add_argument("--exp-pattern", default="*hillas_14_7.0fix.csv")
    ap.add_argument("--sums", default=None, help="sums/edist file (energies in eV)")
    ap.add_argument("--throw-radius", type=float, default=1000.0)
    ap.add_argument("--eff-bins", type=int, default=20)
    ap.add_argument("--unfold", action="store_true")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau-auto", action="store_true")
    ap.add_argument("--emin", type=float, default=1.0)
    ap.add_argument("--emax", type=float, default=200.0)
    ap.add_argument("--energy-bins", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-dist", action="store_true", help="disable dist feature")
    ap.add_argument("--no-wl", action="store_true",   help="disable width/length features")
    ap.add_argument("--no-npix", action="store_true", help="disable numb_pix feature")
    ap.add_argument("--dump", action="store_true", help="dump EXP with reconstructed_energy")
    args = ap.parse_args()

    # load data
    loader = GammaDataLoader(args.model_path, args.exp_path)
    #model_df = loader.load_model_data()
    # TODO fix this hardcode
    model_df_599 = loader.load_model_data(path = '/home/bzzkv/data/599_old_cone_iact1',
                                      pattern = '*iact01*.csv') # load model data (take path with .csv fits)
    model_df_607 = loader.load_model_data(path = '/home/bzzkv/data/607_old_cone_iact1',
                                      pattern = '*iact01*.csv') # load model data (take path with .csv fits)
    model_df = pd.concat([model_df_599, model_df_607])
    
    exp_df   = loader.load_experiment_data(args.exp_pattern)

    # energy bins (ONE source of truth)
    edges, centers = fixed_energy_bins(n_bins=args.energy_bins, e_min_tev=args.emin, e_max_tev=args.emax)

    # keep «reconstructor» for cuts and geo utils
    recon = GammaSpectrumReconstructor(model_df, energy_bins=(edges, centers))
    #model_df = recon.filter_model_data(model_df)
    exp_df   = recon.apply_good_cut(exp_df)
    exp_df   = recon.add_background_alphas_dists_thetas(exp_df)
    exp_df   = recon.classify_by_cuts(exp_df)

    # --- train NN regressor on MODEL ---
    nnr = GammaNNRegressor(
        use_dist = (not args.no_dist),
        use_wl   = (not args.no_wl),
        use_npix = (not args.no_npix),
        model_path="outputs/gamma_regressor.pth",
    )
    os.makedirs("fit_plots", exist_ok=True)
    nnr.fit_on_model(model_df, epochs=args.epochs, batch_size=args.batch_size, out_dir="fit_plots", verbose=True)

    #model_df = recon.filter_model_data(model_df)
    

    # --- predict energy for EXP and MODEL (for R/unfolding) ---
    exp_df["reconstructed_energy"] = nnr.predict_for_df(exp_df, model_like=False)
    model_df_reco = model_df.copy()
    model_df_reco["reconstructed_energy"] = nnr.predict_for_df(model_df_reco, model_like=True)

    # --- A_eff (optional) ---
    eff_csv = os.path.join("outputs", "effective_area.csv")
    if args.sums:
        try:
            from iact_tools.effective_area import compute_effective_area_from_files, write_effective_area_products
            eff_df = compute_effective_area_from_files(
                args.sums,
                model_df,
                n_bins=args.eff_bins,
                radius_m=args.throw_radius,
                energy_col="energy",
                energy_range=[1,200],
                model_filter=None,#GammaSpectrumReconstructor.filter_model_data,
            )
            os.makedirs("outputs", exist_ok=True)
            write_effective_area_products(eff_df, out_dir="outputs", basename="effective_area")
            print(f"[A_eff] saved to outputs/effective_area.*")
        except Exception as e:
            print(f"[A_eff] failed: {e}")

    # --- Optional: Unfolding ---
    if args.unfold:
        try:
            #from iact_tools.reconstruction import GammaSpectrumReconstructor
            from iact_tools.unfolding import run_unfolding_pipeline
            # Ensure reconstructed energies exist for both exp and model
            #model_df_reco = recon.reconstruct_energy_for_model_data(model_df)
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


    # --- Physical spectrum (if calculated A_eff) ---
    if os.path.exists(eff_csv):
        try:
            from iact_tools.analysis import compute_physical_spectrum, plot_physical_spectrum, compute_observation_time_minutes
            eff_df = pd.read_csv(eff_csv)
            exp_gamma = exp_df[exp_df['gamma_like_1'] == 1]
            spec_df = compute_physical_spectrum(exp_gamma, eff_df, energy_col_reco="reconstructed_energy", theta2_col="theta2_1", theta2_max=0.05)
            os.makedirs("outputs", exist_ok=True)
            spec_csv = os.path.join("outputs", "physical_spectrum.csv")
            spec_df.to_csv(spec_csv, index=False)
            plot_physical_spectrum(spec_df, out_png="plots/physical_spectrum.png", logy=True)
            T_min = compute_observation_time_minutes(exp_df, "por", portion_duration_min=2.0)
            print(f"[Spectrum] saved physical_spectrum.csv, T≈{T_min:.1f} min")
        except Exception as e:
            print(f"[Spectrum] failed: {e}")

    # Diagnostics
    analyzer = SpectrumAnalyzer(exp_df)
    analyzer.plot_spectrum()
    analyzer.plot_theta2_distribution()

    if args.dump:
        os.makedirs("outputs", exist_ok=True)
        exp_df.to_csv("outputs/exp_with_energy_nn.csv", index=False)
        print("Saved: outputs/exp_with_energy_nn.csv")

if __name__ == "__main__":
    main()
