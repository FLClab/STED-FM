
import os 

USER = os.environ.get("USER")

if USER == "frbea320":
    BASE_PATH = "/home/frbea320/scratch"
elif USER == "anbil106":
    BASE_PATH = "/home/anbil106/scratch/projects/SSL"
elif USER == "anthony":
    BASE_PATH = "/home-local2/projects/SSL"
elif USER == "frederic":
    BASE_PATH = "/home-local/Frederic"
else:
    raise ValueError("Please set the correct path for the user. Path can be modified in `flc-dataset/experiments/DEFAULTS.py`")

COLORS = {
    "IMAGENET1K_V1": "#5F4690",
    "ImageNet": "#5F4690",
    "JUMP": "#1D6996",
    "HPA": "#0F8554",
    "SIM": "#EDAD08",
    "STED": "#CC503E",
    "Hybrid": "#94346E",
}