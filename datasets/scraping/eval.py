
import os
import random 

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from tqdm.auto import tqdm
from matplotlib import pyplot
from datetime import datetime
from collections import defaultdict

def mask_prompts(example):
    seq = example["text"]
    seq = seq.split(" ")
    protein = seq[-2]
    seq[-2] = "<mask>"
    seq = " ".join(seq)

    return {
        "text" : seq,
        "truth" : protein
    }

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    args = parser.parse_args()

    now = datetime.now().strftime("%Y%m%d%H%M%S")

    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Inference 
    dataset = load_dataset("text", data_files={"validation": "validation.txt"})
    dataset = dataset.map(mask_prompts)

    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)

    confusion_matrices = defaultdict(list)
    for out, truth in zip(tqdm(pipe(KeyDataset(dataset["validation"], key="text"), top_k=1)), KeyDataset(dataset["validation"], key="truth")):

        prediction = out[0]["token_str"]
        confusion_matrices[truth].append(prediction)
        # print(out[0]["sequence"])
        # print(out[0]["score"], out[0]["token_str"])

    avg_accuracy = 0
    for key, values in confusion_matrices.items():
        accuracy = sum([value == key for value in values]) / len(values)
        print(f"{key} : {accuracy:0.2f} ({len(values)} samples)")
        if accuracy < 0.75:
            print(values)
        avg_accuracy += accuracy
    print("Average accuracy: {:0.2f}".format(avg_accuracy / len(confusion_matrices)))

    # fig, ax = pyplot.subplots(figsize=(3,3))
    # ax.hist(scores, range=(0,1), bins=20, histtype="step", color="black")
    # ax.set(
    #     xlabel="Score", ylabel="Frequency"
    # )
    # pyplot.savefig("./results/histogram-{}.png".format(now), bbox_inches="tight", dpi=300)