
import torch
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

model = AutoModelForMaskedLM.from_pretrained("./data/filename-converter/checkpoint-10000")
tokenizer = AutoTokenizer.from_pretrained("./data/filename-converter/checkpoint-10000")

possible_options = list(sorted([
    'beta2spectrin', 'adducin', 'vgat', 'psd95', 'glur1', 'livetubulin', 'bassoon',
    'map2', 'gephyrin', 'fus', 'sirtubulin', 'siractin', 'vglut2', 'vglut1',
    'betacamkii', 'alphacamkii', 'vimentin', 'factin', 'glun2b',
    'nr2b', 'rim', 'tubulin', 'vgat', 'pt286', 'tom20', 'nup',
    'slitrk2', 'smi31'
]))

embeddings = {}
for opt in possible_options:
    encoded = tokenizer.encode(opt)
    embedding = model.roberta.embeddings.word_embeddings.weight[encoded[1]]
    embeddings[opt] = embedding.tolist()

json.dump(embeddings, open("embeddings.json", "w"), indent=4)