import numpy as np 
from typing import List, Dict

class Annotation:
    def __init__(self, filename: str, image: np.ndarray, predictions: Dict[np.ndarray]):
        self.choices = ["ImageNet", "HPA", "JUMP", "STED"]
        assert list(predictions.keys()) == self.choices 
        self.predictions = predictions
        self.annotation = {}

    def check_complete(self):
        return len(set(self.annotation.keys())) == 4
    
    def sample_pair(self):
        if len(self.annotation.keys()) == 0:
            keys = np.random.choice(self.choices, 2)

        else:
            k1 = np.random.choice(list(self.annotation.keys()))
            choices = list(set(self.choices) - set(list(self.annotation.keys())))
            k2 = np.random.choice(choices)
            keys = [k1, k2]

        img1, img2 = [self.predictions[key] for key in keys]
        return (img1, img2), keys 

    def add(self, keys: List[str], choice: str):
        if len(self.annotations.keys()) == 0:
            self.annotation[0] = choice
            keys.remove(choice)
            assert len(keys) == 1
            self.annotation[1] = keys[0]
        else:
            pass
            #TODO
