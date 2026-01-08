import os
import re
import math
import time
import shutil
import hashlib
import hydra
import torch
import logging
import warnings
import functools
import numpy as np
import pytorch_lightning as pl
import multiprocessing as mp
import mrlfads.paths as path

from itertools import product
from distutils.dir_util import copy_tree
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from .utils import replace_hps_str, flatten_params, find_directories, extract_numbers_after_equal, _dir_matches_overrides

# Resolvers for reading config files
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("relpath", lambda p: Path(__file__).parent / ".." / p)

def run(
    config_path: str,              # absolute path for the main config file
    train: bool = True,            # train the model
    nested: bool = False,          # if checkpoint dirs are nested, typically happens for hparam searches
    checkpoint_dir: str = None,    # directory that stores checkpoints
    overrides: dict = {},          # hparam overrides
    use_best: bool = False,
):  
    """Instantiate and execute a PyTorch Lightning experiment from configuration.

    This function reads one or more YAML configuration files to construct the
    PyTorch Lightning model, data module, and callbacks, and then optionally
    runs training.
    """
    # Derive relative path
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    cfg_dir = config_file.parent          # absolute directory containing YAMLs
    cfg_name = config_file.stem           # filename without extension, e.g. "main" for main.yaml

    # Compose the main config from that directory
    overrides_list = [f"{k}={v}" for k, v in flatten_params(overrides).items()]
    with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
        config = compose(config_name=cfg_name, overrides=overrides_list)
        
    # Print
    print('Config path: ', config_path)
    print('Checkpoint path: ', checkpoint_dir)

    # Copy all config files (only do so in `train` mode)
    if train:
        # Collect hydra metadata to access config file paths
        metadata = {}
        with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
            cfg = compose(
                config_name=cfg_name,
                return_hydra_config=True  # ensure cfg.hydra exists
            )
            HydraConfig().set_config(cfg)
            hydra_cfg = HydraConfig.get()
            metadata.update(OmegaConf.to_container(hydra_cfg.runtime.choices))

        # Copy config files into result folder (relative to current working dir)
        os.makedirs("./configs", exist_ok=True)

        for folder in metadata:
            if "hydra" not in folder:
                source_path = os.path.join(str(cfg_dir), folder, metadata[folder] + ".yaml")
                destination_path = os.path.join(".", "configs", folder)
                os.makedirs(destination_path, exist_ok=True)
                shutil.copy(source_path, destination_path)

        # Copy the primary config file itself
        shutil.copy(os.path.join(str(cfg_dir), config_file.name), "./configs")

    # Seed and instantiate datamodule/model
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    datamodule = instantiate(config.datamodule, _convert_="all")
    model = instantiate(config.model)

    # Helper to pick a unique base directory from patterns (for nested checkpoints)
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

        import pdb; pdb.set_trace()
        return base_dirs

    # If a checkpoint directory is provided, locate the most recent checkpoint
    ckpt_path = None
    if checkpoint_dir:
        if nested:
            assert overrides_list != [], "Nested directories require parameter overrides as the pattern finder."
            patterns = [replace_hps_str(override) for override in overrides_list]
            base_dirs = get_base_dirs(patterns)
            base_dir = base_dirs[0]
        else:
            base_dir = str(checkpoint_dir)

        ckpt_pattern = os.path.join(base_dir, "lightning_checkpoints", "*.ckpt")
        candidates = glob(ckpt_pattern)
        ckpt_path = None #next((p for p in candidates if p.endswith("last.ckpt")), None)
        if ckpt_path is None:
            ckpt_path = max(candidates, key=os.path.getctime)

    # Training flow
    if train:
        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
            gradient_clip_val=0.5,
            gradient_clip_algorithm="value",
        )

        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if checkpoint_dir else None,
        )

    # Evaluation-only flow: restore from checkpoint and return artifacts
    elif checkpoint_dir:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_best:
            ckpt = torch.load(trainer.checkpoint_callback.best_model_path, map_location=device)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            
        model.load_state_dict(ckpt["state_dict"])
        return model, datamodule, ckpt

    else:
        return None

def load(
    config_path: str,
    validate: bool=True,
    use_best: bool=False,
    to_scratch: bool=False,
):
    """Load a trained PyTorch Lightning model, datamodule and callbacks from configuration."""
    # Get config path
    CUR_DIR = os.getcwd()
    config_path = Path(config_path)
    run_dir = str(config_path.parent.parent)
    foldername = config_path.parent.parent.name
    
    # Copy config to results to avoid read-only issues specific to CodeOcean
    if to_scratch:
        new_dir = os.path.join(path.homepath, 'scratch', foldername)
        copy_tree(run_dir, new_dir)
        run_dir = new_dir
        config_path = Path(run_dir) / '/'.join(config_path.parts[-2:])
    else:
        pass

    # Switch to the `RUN_DIR` and load the model from checkpoint
    os.chdir(run_dir)
    model, datamodule, ckpt = run(
        str(config_path),
        train = False,
        checkpoint_dir = run_dir,
        use_best = use_best,
    )
    os.chdir(CUR_DIR)

    # Get trainer 
    os.makedirs("lightning_logs", exist_ok=True)
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu', gpus=1)
    else:
        trainer = pl.Trainer(accelerator='cpu')
    model.eval()

    # Run validation using the loaded trainer
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

def load_from_hpsearch(
    run_dir: str,
    rel_config_path: str,
    overrides: dict,
    validate: bool=True,
    use_best: bool=False,
    to_scratch: bool=False,
    specified_hps: list = [],
    derived_hps: list = [],
):
    # Get config path
    CUR_DIR = os.getcwd()
    run_dir = Path(run_dir)
    
    # Copy config to results to avoid read-only issues specific to CodeOcean
    foldername = run_dir.name
    if to_scratch:
        new_dir = os.path.join(path.homepath, 'scratch', foldername)
        copy_tree(str(run_dir), new_dir)
        run_dir = Path(new_dir)
    else:
        pass

    # Separate out 'specified' versus 'derived' hps
    if len(specified_hps) == 0: specified_hps = list(overrides.keys())
    if len(derived_hps) == 0: derived_hps = list(overrides.keys())

    temp = {k: overrides[k] for k in specified_hps}
    candidates = [
        sub for sub in run_dir.iterdir()
        if sub.is_dir() and _dir_matches_overrides(sub.name, temp)
    ]
    
    if not candidates: import pdb; pdb.set_trace()
    elif len(candidates) > 1:
        import pdb; pdb.set_trace()
        
    sub = candidates[0]
    config_path = run_dir / sub / rel_config_path

    # Switch to the `RUN_DIR` and load the model from checkpoint
    os.chdir(run_dir)
    model, datamodule, ckpt = run(
        str(config_path),
        overrides = {k:v for k, v in overrides.items() if k in derived_hps},
        train = False,
        checkpoint_dir = str(run_dir / sub),
        use_best = use_best,
    )
    os.chdir(CUR_DIR)

    # Get trainer 
    try:
        os.makedirs("lightning_logs", exist_ok=True) # unclear if I can just bypass this
    except:
        pass
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu', gpus=1)
    else:
        trainer = pl.Trainer(accelerator='cpu')
    model.eval()

    # Run validation using the loaded trainer
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