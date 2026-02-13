import yaml
from train import run_experiment
import torch
import argparse
seed = 42  # Example seed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with different weight modes.")
    parser.add_argument("--weight_mode", type=str, required=True, help="Specify the weight mode to use.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--model_name", type=str, default="dualpinn", help="Model name to run experiments with.")
    parser.add_argument("--alignment_mode", type=str, default="DERL", help="Alignment mode to use.")
    args = parser.parse_args()
    weight_mode = args.weight_mode
    device = args.device
    model_name = args.model_name
    alignment_mode = args.alignment_mode

   
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