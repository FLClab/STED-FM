
import os 

USER = os.environ.get("USER")

if USER == "frbea320":
    BASE_PATH = "/home/frbea320/scratch"
elif USER == "anbil106":
    BASE_PATH = "/home/anbil106/scratch/projects/SSL"
elif USER == "anthony":
    BASE_PATH = "/home-local2/projects/SSL"
elif USER == "frederic":
    BASE_PATH = "/home-local/Frederic/"
elif USER == "frbea320@ulaval.ca":
    BASE_PATH = "/home/ulaval.ca/frbea320/scratch"
else:
    raise ValueError("Please set the correct path for the user. Path can be modified in `flc-dataset/experiments/DEFAULTS.py`")

# COLORS = {
#     "IMAGENET1K_V1": "#5F4690",
#     "ImageNet": "#5F4690",
#     "JUMP": "#1D6996",
#     "HPA": "#0F8554",
#     "SIM": "#EDAD08",
#     "STED": "#CC503E",
#     "Hybrid": "#94346E",
# }

class ColorMapper:
    def __init__(self):
        if hasattr(self, "__annotations__"):
            for key, value in self.__annotations__.items():
                setattr(self, key, getattr(self, key))
        
    def __getitem__(self, key):
        if not hasattr(self, key):
            # Maybe this is of the form "MAE_SMALL_IMAGENET1K_V1"
            for k in self.__dict__.keys():
                if k.lower() in key.lower():
                    return getattr(self, k)
            return self.default
        return getattr(self, key.lower())
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

class DefaultColorMapper(ColorMapper):

    default: str = "tab:blue"
    imagenet1k_v1: str = "#5F4690"
    imagenet: str = "#5F4690"
    jump: str = "#1D6996"
    hpa: str = "#0F8554"
    sim: str = "#EDAD08"
    sted: str = "#CC503E"
    hybrid: str = "#94346E"
    
COLORS = DefaultColorMapper()
COLORS = {
    "IMAGENET1K_V1": "#5F4690",
    "ImageNet": "#5F4690",
    "JUMP": "#1D6996",
    "HPA": "#0F8554",
    "SIM": "#EDAD08",
    "STED": "#CC503E",
    "Hybrid": "#94346E",
}

MARKERS = {
    "IMAGENET1K_V1": "o",
    "ImageNet": "*",
    "JUMP": "P",
    "HPA": "s",
    "SIM": "^",
    "STED": "o",
}
