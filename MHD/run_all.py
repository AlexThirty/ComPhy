import yaml
from train import run_experiment
import torch
import argparse
seed = 42  # Example seed
if __name__ == "__main__":
    model_name = "dualpinn"
    alignment_modes = ["DERL", "SOB", "OUTL"]
    parser = argparse.ArgumentParser(description="Run experiments with different weight modes.")
    parser.add_argument("--weight_mode", type=str, required=True, help="Specify the weight mode to use.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    args = parser.parse_args()
    weight_modes = [args.weight_mode]
    device = args.device
    
    for alignment_mode in alignment_modes:
        for weight_mode in weight_modes:
            config_path = f"configs/{model_name}.yaml"

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            config["model_config"].update({
                "alignment_mode": alignment_mode,
                "weight_mode": weight_mode,
                "device": device
            })

            print(f"Running experiment with model: {model_name}, alignment mode: {alignment_mode}, weight mode: {weight_mode}")
            run_experiment(
                seed=seed,
                config=config
            )
    
    
    model_names = ["ncl", "pinn"]
    alignment_modes = ["none"]

    for model_name in model_names:
        for alignment_mode in alignment_modes:
            for weight_mode in weight_modes:
                config_path = f"configs/{model_name}.yaml"

                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                config["model_config"].update({
                    "alignment_mode": alignment_mode,
                    "weight_mode": weight_mode,
                    "device": device
                })

                print(f"Running experiment with model: {model_name}, alignment mode: {alignment_mode}, weight mode: {weight_mode}")
                run_experiment(
                    seed=seed,
                    config=config
                )
    
    
    model_name = "pinnncl"
    alignment_modes = ["DERL", "SOB", "OUTL"]

    for alignment_mode in alignment_modes:
        for weight_mode in weight_modes:
            config_path = f"configs/{model_name}.yaml"

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            config["model_config"].update({
                "alignment_mode": alignment_mode,
                "weight_mode": weight_mode,
                "device": device
            })

            print(f"Running experiment with model: {model_name}, alignment mode: {alignment_mode}, weight mode: {weight_mode}")
            run_experiment(
                seed=seed,
                config=config
            )
    