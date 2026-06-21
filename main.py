"""SIMAX Experiment Runner."""

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import load_yaml, setup_logger, ensure_dir
from src.services import ensure_services

logger = setup_logger("main")


def main():
    parser = argparse.ArgumentParser(description="SIMAX Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yml")
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment.yml")
    parser.add_argument("--exp_id", type=str, required=True, help="Unique experiment ID")
    parser.add_argument("--step", type=str, required=True, 
                        choices=["role", "dialogue", "audio", "wet"],
                        help="Experiment step to run")
    parser.add_argument("--port", type=int, default=8000, help="Port for vLLM service")
    
    args = parser.parse_args()
    
    logger.info(f"Running SIMAX | Exp ID: {args.exp_id} | Step: {args.step}")
    
    config = load_yaml(args.config)
    
    # Override port if provided
    if args.port != 8000:
        base_url = config.get("models", {}).get("gpt", {}).get("base_url", "http://localhost:8000/v1")
        if "://" in base_url:
            protocol, rest = base_url.split("://")
            host = rest.split(":")[0]
            new_base_url = f"{protocol}://{host}:{args.port}/v1"
        else:
            new_base_url = f"http://localhost:{args.port}/v1"
        
        if "models" not in config:
            config["models"] = {}
        if "gpt" not in config["models"]:
            config["models"]["gpt"] = {}
        config["models"]["gpt"]["base_url"] = new_base_url
        logger.info(f"Overriding vLLM port to {args.port}. New base_url: {new_base_url}")

    experiment = load_yaml(args.experiment)
    
    results_dir = setup_experiment_dir(args.exp_id, args.config, args.experiment, config)
    
    ensure_services(config, args.step)
    
    execute_step(args.step, config, experiment, args.exp_id, results_dir)
    
    logger.info(f"Step {args.step} completed successfully.")


def setup_experiment_dir(exp_id: str, config_path: str, experiment_path: str, config: dict) -> Path:
    """Setup experiment directory and copy config files."""
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results")) / exp_id
    ensure_dir(results_dir)
    
    shutil.copy(config_path, results_dir / "config.yml")
    shutil.copy(experiment_path, results_dir / "experiment.yml")
    
    logger.info(f"Experiment directory: {results_dir}")
    return results_dir


def execute_step(step: str, config: dict, experiment: dict, exp_id: str, results_dir: Path):
    """Execute the specified step."""
    if step == "role":
        from src.text import role_doc
        role_doc.run(config, experiment, exp_id, results_dir)
    
    elif step == "dialogue":
        from src.text import dialogue_gen
        dialogue_gen.run(config, experiment, exp_id, results_dir)
    
    elif step == "audio":
        from src.audio import preprocess, synthesize
        preprocess.run(config, experiment, exp_id, results_dir)
        synthesize.run(config, experiment, exp_id, results_dir)
    
    elif step == "wet":
        from src.audio import wet
        wet.run(config, experiment, exp_id, results_dir)
    
    else:
        raise ValueError(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
