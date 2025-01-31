from datasets import get_dataset 
import sys 
sys.path.insert(0, "..")
from model_builder import get_base_model




if __name__=="__main__":
    backbone, cfg = get_base_model("mae-lightning-small")
    training_dataset, validation_dataset, testing_dataset = get_dataset(name="factin", cfg=cfg)
    print(f"Factin: {len(training_dataset)}")

    training_dataset, validation_dataset, testing_dataset = get_dataset(name="footprocess", cfg=cfg)
    print(f"Footprocess: {len(training_dataset)}")

    training_dataset, validation_dataset, testing_dataset = get_dataset(name="lioness", cfg=cfg)
    print(f"Lioness: {len(training_dataset)}")


