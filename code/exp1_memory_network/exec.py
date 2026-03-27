import os
import shutil
import mrlfads.paths as path
from pathlib import Path
from datetime import datetime
from mrlfads.run import run

# ---------- USER DEFINED PARAMETERS -----------
PROJECT_STR = os.path.basename(os.getcwd())
EXIST_PROJECT_STR = None
OVERWRITE = isinstance(EXIST_PROJECT_STR, type(None))
# ----------------------------------------------

# Name RUN directory
RUN_STR = PROJECT_STR + '_id' + datetime.now().strftime("%y%m%d%H%M") if not EXIST_PROJECT_STR else EXIST_PROJECT_STR
RUN_DIR = Path(os.path.join(path.homepath, "runs")) / RUN_STR

# Overwrite the directory and re-create new one
if OVERWRITE:
    if RUN_DIR.exists(): shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
else:
    print('WARNING: RUN_DIR NOT OVERWRITTEN.')

# Copy this script
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

# Train the model
os.chdir(RUN_DIR)
run(
    config_path = f'/root/capsule/code/{PROJECT_STR}/main.yaml',
    checkpoint_dir = str(RUN_DIR) if EXIST_PROJECT_STR else None,
    train = True,
)