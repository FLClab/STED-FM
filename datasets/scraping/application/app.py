
from flask import Flask, render_template, request, Response
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from datasets import load_dataset
from tqdm.auto import trange

app = Flask(__name__)

model = AutoModelForMaskedLM.from_pretrained("../data/filename-converter/checkpoint-10000")
tokenizer = AutoTokenizer.from_pretrained("../data/filename-converter/checkpoint-10000")
dataset = load_dataset("text", data_files={"test": "../testing.txt"})
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)

text_prompts = [
    data["text"] for data in dataset["test"]
]

text_prompts = text_prompts
possible_options = list(sorted([
    'beta2spectrin', 'adducin', 'vgat', 'psd95', 'glur1', 'livetubulin', 'bassoon',
    'map2', 'gephyrin', 'fus', 'sirtubulin', 'siractin', 'vglut2', 'vglut1',
    'betacamkii', 'alphacamkii', 'vimentin', 'factin', 'glun2b',
    'nr2b', 'rim', 'tubulin', 'vgat', 'pt286', 'tom20', 'nup'
]))
possible_options += ["unknown"]

class PredictorQueue():
    def __init__(self, text_prompts, block : int = 100, max_items_per_block : float = 0.1) -> None:
        
        self.text_prompts = text_prompts
        self.block = block
        self.max_items_per_block = int(max_items_per_block * self.block)

        self.current_idx = 0
        self.seen_text_prompts = []
        self.queue = self.next_queue()

    def get_text_prompts(self):
        return self.seen_text_prompts

    def next_queue(self):
        queue = []
        for _ in trange(self.block, desc="Computing predictions"):
            prediction = pipe(self.text_prompts[self.current_idx], top_k=1)[0]
            queue.append(prediction)
            self.current_idx += 1

        # List of dictionaries sorted by score
        queue = sorted(queue, key=lambda x: x["score"], reverse=False)
        for item in queue:
            seq = item["sequence"]
            seq = seq.split(" ")
            seq[-2] = "the <mask>"
            seq = " ".join(seq)
            self.seen_text_prompts.append(seq)
        return list(queue)

    def __getitem__(self, idx: int):
        
        # Makes sure that the queue is not empty
        if idx == 0 and len(self.queue) == 0:
            self.queue = self.next_queue()

        # Computes the index in the current queue
        idx = idx % self.max_items_per_block
        if idx == 0:
            self.queue = self.next_queue()

        return self.queue[idx]
        
    def __len__(self):
        return int((len(self.text_prompts) // self.block) * self.max_items_per_block)

prediction_queue = PredictorQueue(text_prompts)
initial_predictions = [None] * len(prediction_queue)
manually_updated = [False] * len(prediction_queue)
predicted_scores = [None] * len(prediction_queue)

current_index = 0

def get_prediction(text_prompt):
    prediction = pipe(text_prompt, top_k=1)[0]
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    global current_index
    global possible_options
    global prediction_queue

    prediction = {"score" : "N/A"}
    if request.method == "POST":

        action = request.form.get("action")
        if action == "prev":
            current_index = max(current_index - 1, 0)
        elif action == "next":
            new_prediction = request.form.get("prediction")
            if new_prediction is not None:
                manually_updated[current_index] = True
                initial_predictions[current_index] = request.form.get("prediction")
            current_index = min(current_index + 1, len(prediction_queue) - 1)
        elif action == "add-option":
            new_option = request.form.get("optionToAdd")
            if new_option is not None and not new_option in possible_options and not new_option == "":
                tmp = possible_options[:-1] + [new_option]
                tmp = list(sorted(tmp))
                possible_options = tmp + possible_options[-1:]

        # Computes the prediction from LLM
        if manually_updated[current_index]:
            prediction = {"score" : "N/A"}
        elif predicted_scores[current_index] is not None:
            prediction = predicted_scores[current_index]
        else:
            prediction = prediction_queue[current_index]
            initial_predictions[current_index] = prediction["token_str"]
            predicted_scores[current_index] = prediction

    with open("results.txt", "w") as f:
        for text_prompt, pred in zip(prediction_queue.get_text_prompts(), initial_predictions):
            if pred is not None:
                # if pred == "unknown":
                #     continue
                text_prompt = text_prompt.replace("<mask>", pred)
                f.write(f"{text_prompt}\n")
            else:
                break
    with open("options.txt", "w") as f:
        for option in possible_options:
            f.write(f"{option}\n")

    return render_template("index.html", 
                           text_prompt=prediction_queue.get_text_prompts()[current_index], 
                           prediction=initial_predictions[current_index], 
                           confidence="{:0.2f}".format(prediction["score"]) if isinstance(prediction["score"], float) else prediction["score"],
                           possible_options=possible_options,
                           current_index=current_index,
                           total_images = len(prediction_queue))

if __name__ == "__main__":
    app.run(debug=True)