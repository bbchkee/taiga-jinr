
#!/usr/bin/env python3
import argparse, os
import pandas as pd
from iact_tools.data_loading import GammaDataLoader
from iact_tools.reconstruction import GammaSpectrumReconstructor
from iact_tools.analysis import SpectrumAnalyzer

def main():
    ap = argparse.ArgumentParser(description="IACT pipeline: load -> fit -> reconstruct -> plots")
    ap.add_argument("--model-path", required=True, help="Path to model CSV folder")
    ap.add_argument("--exp-path", required=True, help="Path to experiment CSV folder")
    ap.add_argument("--exp-pattern", default="*hillas_14_7.0fix.csv", help="Glob for experiment files")
    ap.add_argument("--dump", default=False, action = "store_true", help = "Dump tables with calculated parameters")
    args = ap.parse_args()

    loader = GammaDataLoader(args.model_path, args.exp_path) # init loader
    model_df = loader.load_model_data() # load model data (take path with .csv fits)
    exp_df = loader.load_experiment_data(args.exp_pattern) # load experiment data (take path with .csv fits, you can use regexp for filenames)

    recon = GammaSpectrumReconstructor(model_df) # initializing
    recon.fit_energy_size(out_dir='plots') # calculate dependence size vs energy (linear fit) by model data
    
    exp_df = recon.apply_good_cut(exp_df) # quality cuts (weather, edge, pointing ...)
    exp_df = recon.add_background_alphas_dists_thetas(exp_df) # calculate Hillas geometric pars for background points
    exp_df = recon.classify_by_cuts(exp_df) # add gamma_like flags for source and backgound points. For this is needed to add background alphas dists thetas
    
    exp_df = recon.reconstruct_energy(exp_df) # reconstruct energy
    
    # now we have our df filled with all additional values

    analyzer = SpectrumAnalyzer(exp_df) # init analyzer
    analyzer.plot_spectrum() # plot energy spectrum (at this moment without unfolding and effective area TODO)
    analyzer.plot_theta2_distribution() # plots theta2 excess (source - mean(background))

    if args.dump: # save this huge file do disk if you want
        out_csv = os.path.join("outputs", "exp_with_energy.csv")
        os.makedirs("outputs", exist_ok=True) 
        exp_df.to_csv(out_csv, index=False) 
        print(f"Saved: {out_csv}") 

if __name__ == "__main__":
    main()
