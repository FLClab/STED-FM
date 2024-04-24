
import os
import random 

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from tqdm.auto import tqdm
from matplotlib import pyplot
from datetime import datetime

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    args = parser.parse_args()

    now = datetime.now().strftime("%Y%m%d%H%M%S")

    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Inference 
    dataset = load_dataset("text", data_files={"test": "testing.txt"})
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)

    scores = []
    for out in tqdm(pipe(KeyDataset(dataset["test"], key="text"), top_k=1)):

        # print(out[0]["sequence"])
        # print(out[0]["score"], out[0]["token_str"])
        
        scores.append(out[0]["score"])

    fig, ax = pyplot.subplots(figsize=(3,3))
    ax.hist(scores, range=(0,1), bins=20, histtype="step", color="black")
    ax.set(
        xlabel="Score", ylabel="Frequency"
    )
    pyplot.savefig("./results/histogram-{}.png".format(now), bbox_inches="tight", dpi=300)