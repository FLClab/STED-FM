
import argparse
from typing import Any

class Configuration:
    def __init__(self):
        
        # On the first call, we initialize the dictionary with the default values
        # that are defined in the class.
        if hasattr(self, "__annotations__"):
            for key, value in self.__annotations__.items():
                setattr(self, key, getattr(self, key))

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_dict(self) -> dict:
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configuration):
                out[key] = value.to_dict()
            elif isinstance(value, argparse.Namespace):
                out[key] = value.__dict__
            else:
                out[key] = value
        return out

    def from_dict(self, d) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Configuration().from_dict(v))
            else:
                setattr(self, k, v)
        return self

    @classmethod
    def from_json(cls, path: str) -> None:

        import json
        with open(path, "r") as f:
            d = json.load(f)

        cfg = cls()
        cfg.from_dict(d)

        return cfg

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __copy__(self):
        new = Configuration()
        new.from_dict(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        new = Configuration()
        new.from_dict(self.__dict__)
        return new

    def __reduce__(self):
        return (Configuration, (), self.__dict__)
    
    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        return self.__dict__

    def __setattr__(self, name, value):
        self.__dict__[name] = value
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"Attribute {name} not found in configuration")
    
    def __repr__(self):
        return str(self.__dict__)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--b", type=int, default=2)
    args = parser.parse_args()

    c = Configuration()
    other_configuration = Configuration()
    other_other_configuration = Configuration()
    other_other_configuration.from_dict({"a": 5, "b": 4})
    other_other_configuration.args = args
    other_configuration.from_dict({"a": 6, "b": other_other_configuration})

    c.from_dict({"a": 1, "b": other_configuration})
    print(c.a, c.b.a)

    from copy import copy, deepcopy
    deepc = deepcopy(c)
    
    print(deepc.a, deepc.b.a)
    c.a = 45
    print(deepc.a, deepc.b.a)
    print(c.a, c.b.a)

    from utils import save_cfg
    c.save("tmp.json")

    d = Configuration.from_json("tmp.json")
    d.b.b.b = 10
    d.save("tmp2.json")