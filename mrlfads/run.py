"""
Main entry point for running training and evaluation of MR-LFADS experiments from configuration.
This module provides two main functions:
    - `run()`: Instantiates and executes a PyTorch Lightning experiment based on a provided YAML configuration file. Handles both training and evaluation flows, including checkpoint loading and config management.
    - `load()`: A helper function that loads a trained model and datamodule from a specified config path, and optionally runs validation to return metrics.
"""

import os
import shutil
import torch
import numpy as np
import pytorch_lightning as pl

from glob import glob
from pathlib import Path
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mrlfads.utils.common_utils import replace_hps_str, flatten_params, extract_numbers_after_equal, find_directories

# Resolvers for reading config files
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("relpath", lambda p: Path(__file__).parent / ".." / p)

def run(
    config_path: str, 
    train: bool = True,
    checkpoint_dir: str = None,
    use_best: bool = False,
    nested: bool = False, 
    overrides: dict = {}, 
    checkpoint_override: str = None,
    model_transforms: list = None,
):  
    """
    Instantiate and execute a PyTorch Lightning experiment from configuration.
    Args:
        config_path: Absolute path to the main YAML config file.
        train: Whether to run training or just load the model for evaluation.
        checkpoint_dir: Directory containing checkpoints to restore from. 
            If None, starts training from scratch.
        use_best: If True and checkpoint_dir is provided, loads the best checkpoint according to 
            the trainer's checkpoint callback instead of the most recent one.
        nested: If True, looks for checkpoints in nested directories matching the overrides pattern
            (for ray hyperparameter searches).
        overrides: Dictionary of hyperparameter overrides to apply when composing the config.
        checkpoint_override: If provided, looks for a checkpoint file in checkpoint_dir that contains 
            this string in its name. Useful when multiple checkpoints exist and you want to specify 
            which one to load.
        model_transforms: If provided, applies these transformations to the model after checkpoint
            loading if checkpoint_dir is provided, otherwise after instantiation. Each item should be
            a tuple of (transform_function, kwargs_dict).
    Returns:
        If train is True, returns None. 
        If train is False and checkpoint_dir is provided, returns a tuple of (model, datamodule, 
            checkpoint_dict).
        If train is False and checkpoint_dir is None, returns a tuple of (model, datamodule, None) 
            without loading from checkpoint.
    """

    # Resolve config path and compose config
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    cfg_dir = config_file.parent
    cfg_name = config_file.stem

    overrides_list = [f"{k}={v}" for k, v in flatten_params(overrides).items()]
    with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
        config = compose(config_name=cfg_name, overrides=overrides_list)
        
    print('Config path: ', config_path)
    print('Checkpoint path: ', checkpoint_dir)

    # TRAIN FLOW ONLY:
    # Save a copy of the config files used for this run in a local `configs` directory for reproducibility.
    if train:
        metadata = {}
        with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
            cfg = compose(
                config_name=cfg_name,
                return_hydra_config=True
            )
            HydraConfig().set_config(cfg)
            hydra_cfg = HydraConfig.get()
            metadata.update(OmegaConf.to_container(hydra_cfg.runtime.choices))

        os.makedirs("./configs", exist_ok=True)

        for folder in metadata:
            if "hydra" not in folder:
                source_path = os.path.join(str(cfg_dir), folder, metadata[folder] + ".yaml")
                destination_path = os.path.join(".", "configs", folder)
                os.makedirs(destination_path, exist_ok=True)
                shutil.copy(source_path, destination_path)

        shutil.copy(os.path.join(str(cfg_dir), config_file.name), "./configs")

    # Set random seed for reproducibility
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Instantiate datamodule and model
    datamodule = instantiate(config.datamodule, _convert_="all")
    model = instantiate(config.model)

    def get_base_dirs(patterns):
        base_dirs = find_directories(checkpoint_dir, *patterns)
        if len(base_dirs) == 1:
            return base_dirs

        target_vals = [float(p.split("=", 1)[1]) for p in patterns]
        matching_idxs = []
        for i, d in enumerate(base_dirs):
            dir_vals = extract_numbers_after_equal(d)
            if np.all(np.array(target_vals) == np.array(dir_vals)):
                matching_idxs.append(i)

        if len(matching_idxs) == 1:
            return [base_dirs[matching_idxs[0]]]
        else:
            raise RuntimeError(
                f"Expected exactly one directory matching patterns {patterns} in {checkpoint_dir}, but found {len(base_dirs)} candidates."
            )

    # Determine checkpoint path to load from if checkpoint_dir is provided
    ckpt_path = checkpoint_override
    if checkpoint_dir is not None:
        if nested:
            assert overrides_list != [], "Nested directories require parameter overrides as the pattern finder."
            patterns = [replace_hps_str(override) for override in overrides_list]
            base_dirs = get_base_dirs(patterns)
            base_dir = base_dirs[0]
        else:
            base_dir = str(checkpoint_dir)

        ckpt_pattern = os.path.join(base_dir, "lightning_checkpoints", "*.ckpt")
        candidates = glob(ckpt_pattern)
        if ckpt_path is None:
            ckpt_path = max(candidates, key=os.path.getctime)
        else:
            matches = [p for p in candidates if ckpt_path in os.path.basename(p)]
            if len(matches) == 0:
                raise FileNotFoundError(
                    f"No checkpoint matching pattern '{ckpt_path}' found in {ckpt_pattern}"
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple checkpoints match pattern '{ckpt_path}':\n" +
                    "\n".join(matches)
                )
            ckpt_path = matches[0]

    # TRAIN FLOW ONLY
    if train:
        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            accelerator="gpu",
            devices="auto",
            gradient_clip_val=0.5,
            gradient_clip_algorithm="value",
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if checkpoint_dir else None,
        )

    # EVAL FLOW ONLY
    elif checkpoint_dir:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    else:
        pass

    # Apply custom transforms to the model
    if model_transforms is not None:
        for (transform, kwargs) in model_transforms:
            model = transform(model, **kwargs)
    return model, datamodule, None

def load(
    config_path: str,
    validate: bool=True,
    use_best: bool=False,
):
    """
    Load a trained PyTorch Lightning model, datamodule and callbacks from configuration.
    Args:
        config_path: Absolute path to the main YAML config file.
        validate: Whether to run validation after loading the model. 
            If False, returns the model and datamodule without running validation.
        use_best: If True, loads the best checkpoint according to the trainer's checkpoint callback
            instead of the most recent one.
    Returns:
        Returns a dictionary containing the loaded model, datamodule, trainer, and validation metrics
            (if validate is True), and checkpoint dictionary (if checkpoint_dir is provided in the config).
    """
    # Resolve config path and compose config
    CUR_DIR = os.getcwd()
    config_path = Path(config_path)
    run_dir = str(config_path.parent.parent)

    # Switch to the `RUN_DIR` and load the model from checkpoint
    os.chdir(run_dir)
    model, datamodule, ckpt = run(
        str(config_path),
        train = False,
        checkpoint_dir = run_dir,
        use_best = use_best,
    )
    os.chdir(CUR_DIR)

    # Set up a trainer for validation
    os.makedirs("lightning_logs", exist_ok=True)
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu', devices=1)
    else:
        trainer = pl.Trainer(accelerator='cpu')
    model.eval()

    #  Run validation and extract metrics if requested, otherwise return None for metrics
    if validate:
        trainer.validate(model, datamodule=datamodule, verbose=False)
        metrics = trainer.logged_metrics
        shutil.rmtree("lightning_logs")
    else:
        metrics = None
    
    # Return
    state_dict = {
        "model": model,
        "datamodule": datamodule,
        "trainer": trainer,
        "metrics": metrics,
        "ckpt": ckpt,
    }
    return state_dict

