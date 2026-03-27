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
import torch.nn as nn
import pytorch_lightning as pl

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

import mrlfads.paths as path
from mrlfads.utils.common_utils import replace_hps_str, flatten_params, find_directories, extract_numbers_after_equal, dir_matches_overrides

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
    checkpoint_override: str = None,
    model_overrides: list = None,
):  
    """Instantiate and execute a PyTorch Lightning experiment from configuration."""
    # Assertions
    assert checkpoint_dir is None or model_overrides is None
    
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
        # ckpt_path = next((p for p in candidates if p.endswith("last.ckpt")), None)
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
            
    elif model_overrides is not None:
        for (transform, kwargs) in model_overrides: model = transform(model, **kwargs)

    # Training flow
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
        return {'model': model, 'datamodule': datamodule}

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
        trainer = pl.Trainer(accelerator='gpu', devices=1)
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
    checkpoint_override: str = None,
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
        if sub.is_dir() and dir_matches_overrides(sub.name, temp)
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
        checkpoint_override = checkpoint_override,
    )
    os.chdir(CUR_DIR)

    # Get trainer 
    try:
        os.makedirs("lightning_logs", exist_ok=True) # unclear if I can just bypass this
    except:
        pass
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu', devices="auto",)
    else:
        trainer = pl.Trainer(accelerator='cpu')
    model.eval()

    # Run validation using the loaded trainer
    if validate:
        trainer.validate(model, datamodule=datamodule, verbose=False)
        metrics = trainer.logged_metrics
    else:
        metrics = None
    shutil.rmtree("lightning_logs")
    
    # Return
    state_dict = {
        "model": model,
        "datamodule": datamodule,
        "trainer": trainer,
        "metrics": metrics,
        "ckpt": ckpt,
    }
    return state_dict


# ---------- Copy model weights ---------- #

def transfer_modules(
    new_model,
    load_type: str,
    load_type_kwargs: dict,
    transfer_dict: dict,
    requires_grad_off: bool = False,
):
    # Load model according to load type
    if load_type == 'single':
        old_state_dict = load(**load_type_kwargs)
    elif load_type == 'hpsearch':
        old_state_dict = load_from_hpsearch(**load_type_kwargs)
    else:
        raise ValueError()
        
    old_model = old_state_dict['model']
    
    # Transfer
    for area_name, transfer_comps in transfer_dict.items():
        print(f'Transfering weights of {area_name}...')
        old_area = old_model.areas[area_name]
        new_area = new_model.areas[area_name]

        for comp in transfer_comps:
            old_comp = getattr(old_area, comp, None)
            new_comp = getattr(new_area, comp, None)

            if old_comp is None or new_comp is None:
                raise ValueError(f"component '{comp}' not found on area '{area_name}'")

            transfer_params(old_comp, new_comp, requires_grad_off=requires_grad_off)
    
    return new_model

import torch.nn as nn


def transfer_params(
    old: nn.Module,
    new: nn.Module,
    verbose: bool = True,
    requires_grad_off: bool = False,
):
    old_sd = old.state_dict()
    new_sd = new.state_dict()

    transferred_keys = set()

    for key, old_tensor in old_sd.items():
        if key not in new_sd:
            if verbose:
                print(f"[transfer] '{key}' not present in new component – skipping")
            continue

        new_tensor = new_sd[key]

        if old_tensor.shape == new_tensor.shape:
            new_tensor.copy_(old_tensor)
            transferred_keys.add(key)
            if verbose:
                print(f"[transfer] copied '{key}'")
            continue

        if verbose:
            print(
                f"[transfer] shape mismatch for '{key}': "
                f"{tuple(old_tensor.shape)} vs {tuple(new_tensor.shape)}"
            )

        # Partial front-aligned copy if old fits inside new
        if (
            old_tensor.ndim == new_tensor.ndim
            and all(o <= n for o, n in zip(old_tensor.shape, new_tensor.shape))
        ):
            slices = tuple(slice(0, s) for s in old_tensor.shape)
            new_tensor[slices].copy_(old_tensor)
            transferred_keys.add(key)
            if verbose:
                print(
                    f"[transfer] partially copied '{key}' into leading slice "
                    f"{slices}"
                )
        else:
            if verbose:
                print(f"[transfer] '{key}' does not fit in new tensor – skipping")

    if requires_grad_off:
        for key, param in new.named_parameters():
            if key in transferred_keys:
                param.requires_grad_(False)
                if verbose:
                    print(f"[transfer] turned requires_grad off for '{key}'")
                    
def register_toggle(
    new_model,
    components: list,
    trigger: int,
    verbose: bool= True,
):
    for area in new_model.areas.values():
        area.registered_funcs['toggle'] = (trigger, {'components': components, 'requires_grad': True, 'verbose': verbose})
    return new_model