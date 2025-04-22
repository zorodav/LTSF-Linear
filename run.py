import argparse
import yaml
import datetime
from pathlib import Path
import torch
from exp.exp_main import Exp_Main
import numpy as np
import matplotlib.pyplot as plt

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration details
    print(config)
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(f"results/{dataset_name}/{model_name}/{batch_id}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration to results directory
    with open(results_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Initialize experiment
    exp = Exp_Main(config)

    # Training and evaluation loop
    batch_results = {}
    if config['training']['is_training']:
        for itr in range(config['training']['iterations']):
            setting = f"{model_name}_{dataset_name}_itr{itr}"
            print(f">>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            if not config['training']['train_only']:
                print(f">>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                metrics = exp.test(setting)
                batch_results[f"iteration_{itr}"] = metrics

            if config['training']['do_predict']:
                print(f">>>>>>> Predicting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                predictions, metrics = exp.predict(setting, save_results=True)
                pair_dir = results_dir / f"iteration_{itr}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                np.save(pair_dir / 'predictions.npy', predictions)
                with open(pair_dir / 'evaluation_results.yaml', 'w') as f:
                    yaml.dump(metrics, f)

                # Save visualizations
                vis_dir = pair_dir / 'visualizations'
                vis_dir.mkdir(parents=True, exist_ok=True)
                exp.save_visualizations(vis_dir)

            torch.cuda.empty_cache()
    else:
        setting = f"{model_name}_{dataset_name}_test"
        if config['training']['do_predict']:
            print(f">>>>>>> Predicting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            predictions, metrics = exp.predict(setting, save_results=True)
            pair_dir = results_dir / "test"
            pair_dir.mkdir(parents=True, exist_ok=True)
            np.save(pair_dir / 'predictions.npy', predictions)
            with open(pair_dir / 'evaluation_results.yaml', 'w') as f:
                yaml.dump(metrics, f)

            # Save visualizations
            vis_dir = pair_dir / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)
            exp.save_visualizations(vis_dir)
        else:
            print(f">>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            metrics = exp.test(setting, test=True)
            batch_results["test"] = metrics

        torch.cuda.empty_cache()

    # Save batch results summary
    with open(results_dir / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
