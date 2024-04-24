import torch
import math
import os 
import glob
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline, TrainerCallback
from datasets import load_dataset

from datasets import ClassLabel
import random
import pandas as pd
from transformers.trainer_callback import TrainerControl, TrainerState

class TestMaskingCallback(TrainerCallback):

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        
        checkpoints = glob.glob(os.path.join(args.output_dir, 'checkpoint-*'))
        if len(checkpoints) == 0:
            return super().on_evaluate(args, state, control, **kwargs)
        
        last_ckpt = max([int(ckpt.split('-')[-1]) for ckpt in checkpoints])
        if os.path.exists(os.path.join(args.output_dir, f'checkpoint-{last_ckpt}')):
            print("Loading model from checkpoint: ", f"checkpoint-{last_ckpt}")
            model = AutoModelForMaskedLM.from_pretrained(os.path.join(args.output_dir, f'checkpoint-{last_ckpt}'))
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.output_dir, f'checkpoint-{last_ckpt}'))

            # Inference 
            # text = "Channel STED 594 in filename 'InfDIV14 PLKO MOI10 11DPI VGLUT1488 MAP2490LS FUS594 PSD955635 dendrite linksoma10 img10.msr' is for the <mask> protein.".lower()
            with open("validation.txt", "r") as f:
                lines = f.readlines()
                idx = random.randint(0, len(lines)-1)
                text = lines[idx].strip().lower()
                tokens = text.split(" ")
                tokens[-2] = "<mask>"
                text = " ".join(tokens)
            mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer)
            print(mask_filler(text, top_k=3))

        return super().on_evaluate(args, state, control, **kwargs)
    

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    print(df)

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-from", type=str, default=None)
    args = parser.parse_args()

    datasets = load_dataset("text", data_files={"train": "training.txt", "validation" : "validation.txt"})
    # show_random_elements(datasets["train"])

    if args.restore_from is not None:
        model = AutoModelForMaskedLM.from_pretrained(args.restore_from)
        tokenizer = AutoTokenizer.from_pretrained(args.restore_from)
    else:
        model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')


    num_added_tokens = 0
    num_added_tokens += tokenizer.add_tokens([
        'beta2spectrin', 'adducin', 'vgat', 'psd95', 'glur1', 'livetubulin', 'bassoon',
        'map2', 'gephyrin', 'fus', 'sirtubulin', 'siractin', 'vglut2', 'vglut1',
        'betacamkii', 'alphacamkii', 'vimentin', 'factin', 'glun2b',
        'nr2b', 'rim', 'tubulin', 'vgat', 
        'unknown'
    ])
    for text in datasets["train"]["text"]:
        num_added_tokens += tokenizer.add_tokens(text.split())
    print(f"Added {num_added_tokens} tokens")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    block_size = 256
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        batch_size=2048,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./data/filename-converter",
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=2e-5,
        max_steps=10000,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        callbacks=[TestMaskingCallback()]
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Inference 
    text = "Channel STED 594 in filename 'InfDIV14 PLKO MOI10 11DPI VGLUT1488 MAP2490LS FUS594 PSD955635 dendrite linksoma10 img10.msr' is for the <mask> protein.".lower()
    mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    print(mask_filler(text, top_k=3))

if __name__ == "__main__":

    main()
