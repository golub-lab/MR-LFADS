import os
import ray
import hydra
import shutil
import pytorch_lightning as pl

from pathlib import Path
from datetime import datetime
from ray import train, tune, air
from ray.tune import CLIReporter
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

import mrlfads.paths as path
from mrlfads.run import run

# ---------- USER DEFINED PARAMETERS -----------
PROJECT_STR = os.path.basename(os.getcwd())
EXIST_PROJECT_STR = None
HYPERPARAM_SPACE = {
    "seed": tune.grid_search([0, 1, 2]),
}
OVERWRITE = True
resources_per_trial = {'cpu': 16, 'gpu': 0.2}
# ----------------------------------------------

# Name RUN directory
RUN_STR = PROJECT_STR + '_id' + datetime.now().strftime("%y%m%d%H%M") if not EXIST_PROJECT_STR else EXIST_PROJECT_STR
RUN_DIR = Path(os.path.join(path.resultpath)) / RUN_STR

# Overwrite the directory and re-create new one
if OVERWRITE:
    if RUN_DIR.exists(): shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
else:
    print('WARNING: RUN_DIR NOT OVERWRITTEN.')

# Copy this script
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

# Load initial configuration
os.chdir(RUN_DIR)
config_path = os.path.join(CUR_DIR, 'main.yaml')
config_file = Path(config_path).expanduser()

if not config_file.is_absolute():
    config_file = (Path.cwd() / config_file).resolve()
cfg_dir = config_file.parent          # absolute directory containing YAMLs
cfg_name = config_file.stem           # filename without extension, e.g. "main" for main.yaml

with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
    config = compose(config_name=cfg_name)

# Define trainable function
def trainable(overrides, config):
    trial_dir = train.get_context().get_trial_dir()
    os.chdir(trial_dir)
    loss = run(config_path=config_path, overrides=overrides)
    train.report({"loss": loss})

# Run the hyperparameter search
run_config = air.RunConfig(
    storage_path=str(RUN_DIR.parent),
    name=RUN_DIR.name,
    checkpoint_config=air.CheckpointConfig(
        checkpoint_at_end=False,
        checkpoint_frequency=0,
        num_to_keep=1,
    ),
)
tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(trainable, config=config),
        resources=resources_per_trial,
    ),
    param_space=HYPERPARAM_SPACE,
    run_config=run_config,
)
results = tuner.fit()

ray.shutdown()