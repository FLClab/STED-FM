from flask import Flask, render_template, request, Response
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from datasets import load_dataset

app = Flask(__name__)

model = AutoModelForMaskedLM.from_pretrained("../data/test-clm/checkpoint-10000")
tokenizer = AutoTokenizer.from_pretrained("../data/test-clm/checkpoint-10000")
dataset = load_dataset("text", data_files={"test": "../testing.txt"})
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

text_prompts = [
    data["text"] for data in dataset["test"]
]

text_prompts = text_prompts
possible_options = list(sorted([
    'beta2spectrin', 'adducin', 'vgat', 'psd95', 'glur1', 'livetubulin', 'bassoon',
    'map2', 'gephyrin', 'fus', 'sirtubulin', 'siractin', 'vglut2', 'vglut1',
    'betacamkii', 'alphacamkii', 'vimentin', 'factin', 'tubulin'
]))
possible_options += ["unknown"]
initial_predictions = [None] * len(text_prompts)
manually_updated = [False] * len(text_prompts)
predicted_scores = [None] * len(text_prompts)

current_index = 0

def get_prediction(text_prompt):
    prediction = pipe(text_prompt, top_k=1)[0]
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    global current_index
    global possible_options

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
            current_index = min(current_index + 1, len(text_prompts) - 1)
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
            prediction = get_prediction(text_prompts[current_index])
            initial_predictions[current_index] = prediction["token_str"]
            predicted_scores[current_index] = prediction

    with open("results.txt", "w") as f:
        for text_prompt, pred in zip(text_prompts, initial_predictions):
            if pred is not None:
                if pred == "unknown":
                    continue
                text_prompt = text_prompt.replace("<mask>", pred)
                f.write(f"{text_prompt}\n")
            else:
                break
    with open("options.txt", "w") as f:
        for option in possible_options:
            f.write(f"{option}\n")

    return render_template("index.html", 
                           text_prompt=text_prompts[current_index], 
                           prediction=initial_predictions[current_index], 
                           confidence="{:0.2f}".format(prediction["score"]) if isinstance(prediction["score"], float) else prediction["score"],
                           possible_options=possible_options,
                           current_index=current_index,
                           total_images = len(text_prompts))

if __name__ == "__main__":
    app.run(debug=True)