
import os
import glob
import sys
import random
import time
import threading
import pickle
import json 
import uuid

from flask import Flask, render_template, request, redirect, url_for, g, session
from multiprocessing import Manager

import logging

logging.basicConfig()
logger = logging.getLogger('gunicorn.error')
# logger = logging.getLogger("app")
# logger.setLevel(logging.INFO)
print("Logger: {}".format(logger))

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

# Restore users
if not os.path.exists("data"):
    os.makedirs("data")
with open(os.path.join("data", "users.json"), "w") as f:
    json.dump({}, f)

class User:
    def __init__(self, name):
        self.name = name

DATASET = "example"
os.makedirs(os.path.join("data", DATASET), exist_ok=True)

if DATASET in ["example"]:
    template_images = glob.glob(os.path.join("static", DATASET, "templates", "*.png"))
    template_images = [os.path.relpath(path, "static") for path in template_images]

    MODEL_IDS = [
        "model-a",
        "model-b",
    ]

    candidate_images = []
    tmp = []
    for image in template_images:
        basename = os.path.basename(image)
        candidate_images.extend([
            image.replace("templates", "candidates").replace(basename, f"{model_id}_{basename}")
            for model_id in MODEL_IDS
        ])
        tmp.extend([
            image for _ in MODEL_IDS
        ])

    template_images = tmp
    random.seed(42)
    tmp = list(zip(template_images, candidate_images))
    random.shuffle(tmp)

    template_images, candidate_images = zip(*tmp)
    template_images = list(template_images)
    candidate_images = list(candidate_images)
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

# Shuffle the candidate images
print("Number of images: ", len(candidate_images))

@app.before_request
def get_globals():

    global user, current_idx, user_choices

    username = session.get("user", None)
    if username is None:
        user = None
        current_idx = 0
        user_choices = []
    else:
        if os.path.isfile(os.path.join("data", DATASET, f"{username}.pkl")):
            logger.log(logging.INFO, f"Found data for `{username}`")
            try:
                with open(os.path.join("data", DATASET, f"{username}.pkl"), "rb") as f:
                    data = pickle.load(f)
                    user = data["user"]
                    current_idx = data["current_idx"]
                    user_choices = data["user_choices"]
            except EOFError:
                print("NO DATA")
        else:
            user = User(username)
            current_idx = 0
            user_choices = []
            
    if user is not None:
        logger.log(logging.INFO, "PID: {}".format(os.getpid()))
        logger.log(logging.INFO, f"User: {user.name}")
        logger.log(logging.INFO, f"Current index: {current_idx}")
        # logger.log(logging.INFO, f"User choices: {user_choices}")

    return None

@app.after_request
def save_globals(response):

    global user, current_idx, user_choices

    if user is not None:
        data = {
            "user": user,
            "current_idx": current_idx,
            "user_choices": user_choices
        }
        logger.log(logging.INFO, f"[{user.name}] Saving data.")
        with open(os.path.join("data", DATASET, f"{user.name}.pkl"), "wb") as f:
            pickle.dump(data, f)

    return response

@app.route("/done")
def done():
    return render_template('done.html')

@app.route("/login", methods=['POST'])
def login():
    global user, current_idx, user_choices

    user = User(request.form.get('username'))

    if not os.path.isfile(os.path.join("data", "users.json")):
        with open(os.path.join("data", "users.json"), "w") as f:
            json.dump({}, f)
    
    with open(os.path.join("data", "users.json"), "r") as f:
        users = json.load(f)
        # User exists and is connected
        if user.name in users and users[user.name] == 1:
            return render_template('login.html')
        # User exists but is not connected
        elif user.name in users and users[user.name] == 0:
            users[user.name] = 1
        # User does not exist
        else:
            users[user.name] = 1
    with open(os.path.join("data", "users.json"), "w") as f:
        json.dump(users, f)

    logger.log(logging.INFO, "--------------------------------------")
    logger.log(logging.INFO, "User logged in: {}".format(user.name))
    logger.log(logging.INFO, "--------------------------------------")
    
    session['user'] = user.name

    if os.path.isfile(os.path.join("data", DATASET, f"{user.name}.pkl")):
        logger.log(logging.INFO, "Loading data for user: {}".format(user.name))
        with open(os.path.join("data", DATASET, f"{user.name}.pkl"), "rb") as f:
            data = pickle.load(f)
            user = data["user"]
            current_idx = data["current_idx"]
            user_choices = data["user_choices"]

    return redirect(url_for('index'))

@app.route('/')
def index():

    global user, current_idx, candidate_images

    logger.log(logging.INFO, "[INDEX]")
    logger.log(logging.INFO, "User: {}".format(user))
    logger.log(logging.INFO, "Current index: {}; Candidate Images: {}".format(current_idx, len(candidate_images)))

    if user is None:
        return render_template('login.html')

    # When all the images have been annotated, display the done page
    if current_idx >= len(candidate_images):
        logger.log(logging.INFO, "User ({}) is done".format(user))
        return redirect(url_for('done'))

    # Shuffles the candidate images
    tmp = [template_images[current_idx], candidate_images[current_idx]]
    random.shuffle(tmp)
    return render_template('index.html', candidate_images=tmp)

@app.route('/choose', methods=['POST'])
def choose():

    global user, current_idx, template_images, candidate_images, user_choices

    # Get the selected image from the form
    selected_image = request.form.get('selected_image')

    if user is None:
        # Avoids internal server error
        return redirect(url_for('index'))
    
    logger.log(logging.INFO, "User ({}) selected: {}".format(user.name, selected_image))
    
    # Save the choice
    if current_idx < len(candidate_images):
        user_choices.append([template_images[current_idx], candidate_images[current_idx], selected_image])

    current_idx += 1
    print(user_choices, current_idx)

    # Redirect back to the main page after the choice
    return redirect(url_for('index'))

@app.route("/logout", methods=['POST'])
def logout():

    global current_idx, user, candidate_images

    logger.log(logging.INFO, "--------------------------------------")
    logger.log(logging.INFO, "User logged out: {}".format(user.name))
    logger.log(logging.INFO, "--------------------------------------")
    
    with open(os.path.join("data", "users.json"), "r") as f:
        users = json.load(f)
        users[user.name] = 0
    with open(os.path.join("data", "users.json"), "w") as f:
        json.dump(users, f)

    session.pop('user', None)

    user = None
    current_idx = 0

    return redirect(url_for('index'))

if __name__ == '__main__':
    
    app.run(host="0.0.0.0", port=5000)
