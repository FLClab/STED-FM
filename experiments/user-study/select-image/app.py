
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
with open(os.path.join("data", "users.json"), "w") as f:
    json.dump({}, f)

class User:
    def __init__(self, name):
        self.name = name

DATASET = "attention-maps"
os.makedirs(os.path.join("data", DATASET), exist_ok=True)

if DATASET == "attention-maps":

    IMAGE_IDS = [
        "MAE_SMALL_IMAGENET1K_V1",
        "MAE_SMALL_JUMP",
        "MAE_SMALL_HPA",
        "MAE_SMALL_SIM",
        "MAE_SMALL_STED"
    ]

    # Dummy data for images
    template_images = glob.glob(os.path.join("static", DATASET, "templates", "*.png"))
    template_images = [os.path.relpath(path, "static") for path in template_images]
    random.seed(42)
    random.shuffle(template_images)

    candidate_images = []
    for image in template_images:
        basename = os.path.basename(image)
        candidate_images.append([
            image.replace("templates", "candidates").replace(basename, f"{image_id}_{basename}")
            for image_id in IMAGE_IDS
        ])

elif DATASET == "preference-study":

    IMAGE_IDS = [
        "MAE_SMALL_IMAGENET1K_V1",
        "MAE_SMALL_JUMP",
        "MAE_SMALL_HPA",
        "MAE_SMALL_SIM"
        "MAE_SMALL_STED"
    ]

    # Dummy data for images
    template_images = glob.glob(os.path.join("static", DATASET, "templates", "*.png"))
    template_images = [os.path.relpath(path, "static") for path in template_images]
    random.seed(42)
    random.shuffle(template_images)

    candidate_images = []
    for image in template_images:
        basename = os.path.basename(image)

        to_append = [
            image.replace("templates", "candidates").replace(basename, f"{image_id}_{basename}")
            for image_id in IMAGE_IDS
        ]
        if any([os.path.isfile(os.path.join("static", path)) for path in to_append]):
            candidate_images.append(to_append)
        # candidate_images.append([
        #     image.replace("templates", "candidates").replace(basename, f"{image_id}_{basename}")
        #     for image_id in IMAGE_IDS
        # ])

# Shuffle the candidate images
for i in range(len(candidate_images)):
    random.shuffle(candidate_images[i])

@app.before_request
def get_globals():

    global user, current_idx, user_choices

    username = session.get("user", None)
    if username is None:
        user = None
        current_idx = 0
        user_choices = {}
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
            user_choices = {}
            
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
        logger.log(logging.INFO, f"[{user.name}] Saving data: {data}")
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

    return render_template('index.html', template_image=template_images[current_idx], candidate_images=candidate_images[current_idx])

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
        user_choices[template_images[current_idx]] = selected_image

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
