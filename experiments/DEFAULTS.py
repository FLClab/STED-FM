
import os 

USER = os.environ.get("USER")

if USER == "frbea320":
    BASE_PATH = "/home/frbea320/scratch/"
elif USER == "anbil106":
    BASE_PATH = "/home/anbil106/scratch/projects/SSL"
elif USER == "anthony":
    BASE_PATH = "/home-local2/projects/SSL"
else:
    raise ValueError("Please set the correct path for the user. Path can be modified in `flc-dataset/experiments/DEFAULTS.py`")