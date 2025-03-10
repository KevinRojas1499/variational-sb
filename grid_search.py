import subprocess
import json
from pathlib import Path
import itertools

# Grid search parameters
datasets = ['exchange_rate']
damp_coefs = [0.9]
beta_maxs = [8, 10, 12]
seeds = [43, 44, 45, 56, 12342, 1234]

# Create results directory
results_dir = Path("grid_search_results")
results_dir.mkdir(exist_ok=True)

# File to store all results and summaries
results_file = results_dir / "all_results.txt"
with open(results_file, "w") as f:
    f.write("Grid Search Results\n\n")

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    all_results = []
    
    # Try all combinations
    for damp, beta, seed in itertools.product(damp_coefs, beta_maxs, seeds):
        # Create unique directory for this run
        run_dir = f"results/grid_{dataset}_{damp}_{beta}"
        
        # Run the main experiment
        main_cmd = [
            "python3", "time_series/main.py",
            "--data", dataset,
            "--sde", "linear-momentum-sb",
            "--damp_coef", str(damp),
            "--beta_max", str(beta),
            "--seed", str(seed),
            "--dir", run_dir
        ]
        
        plot_cmd = [
            "python3", "time_series/generate_plots.py",
            "--path", run_dir
        ]
        
        print(f"\nRunning experiment with damp={damp}, beta={beta}")
        subprocess.run(main_cmd)
        subprocess.run(plot_cmd)
        
        # Read the metrics file
        metrics_file = Path(run_dir) / "summary_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                current_value = metrics.get("CRPS-Sum", float('inf'))
                all_results.append({
                    "damp_coef": damp,
                    "beta_max": beta,
                    "seed" : seed,
                    "value": current_value
                })
    
    # Write all results and summaries for this dataset
    with open(results_file, "a") as f:
        f.write(f"Results for dataset: {dataset}\n\n")
        f.write("All Results:\n")
        for result in all_results:
            f.write(f"Damp={result['damp_coef']}, Beta={result['beta_max']}: {result['value']}\n")
        
        f.write("\nSummary by Damping Coefficient:\n")
        for damp in damp_coefs:
            damp_results = [r for r in all_results if r['damp_coef'] == damp]
            if damp_results:
                best_for_damp = min(damp_results, key=lambda x: x['value'])
                f.write(f"\nDamping coefficient: {damp}\n")
                f.write(f"Best beta max: {best_for_damp['beta_max']}\n")
                f.write(f"Best seed: {best_for_damp['seed']}\n")
                f.write(f"Best value: {best_for_damp['value']}\n")
        
        f.write("\n" + "-" * 40 + "\n\n")
        
    # Find overall best parameters
    if all_results:
        best_result = min(all_results, key=lambda x: x['value'])
        print(f"\nCompleted {dataset}.")
        print(f"Overall best parameters: Damp={best_result['damp_coef']}, "
              f"Beta={best_result['beta_max']}, Value={best_result['value']}")
