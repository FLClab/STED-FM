
import os 

USER = os.environ.get("USER")

if USER == "frbea320":
    BASE_PATH = "/home/frbea320/projects/def-flavielc/"
elif USER == "anbil106":
    BASE_PATH = "/home/anbil106/scratch/anbil106/SSL"
elif USER == "anthony":
    BASE_PATH = "/home-local2/projects/SSL"
elif USER == "koles2":
    BASE_PATH = "/home/koles2/projects/def-flavielc/"
else:
    raise ValueError("Please set the correct path for the user. Path can be modified in `flc-dataset/experiments/DEFAULTS.py`")