import argparse
import yaml
import datetime
from pathlib import Path
import torch
from exp.exp_main import Exp_Main


def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration details
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize experiment
    exp = Exp_Main(config)

    # Training and evaluation loop
    if config['training']['is_training']:
        for itr in range(config['training']['iterations']):
            setting = f"{model_name}_{dataset_name}_itr{itr}"
            print(f">>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            if not config['training']['train_only']:
                print(f">>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.test(setting)

            if config['training']['do_predict']:
                print(f">>>>>>> Predicting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.predict(setting, save_results=True)

            torch.cuda.empty_cache()
    else:
        setting = f"{model_name}_{dataset_name}_test"
        if config['training']['do_predict']:
            print(f">>>>>>> Predicting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.predict(setting, save_results=True)
        else:
            print(f">>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting, test=True)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
