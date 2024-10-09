
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

gunicorn_logger = logging.getLogger('gunicorn.error')

sys.path.insert(0, "..")
from annotation import build_tree, Queue, Tree

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

# Restore users
with open("users.json", "w") as f:
    json.dump({}, f)

class User:
    def __init__(self, name):
        self.name = name

CLASS_ID = "perforated"
CALLED = 0

# Dummy data for images
template_image = f"{CLASS_ID}/template.png"
candidate_images = glob.glob(os.path.join("static", CLASS_ID, "candidates", "*.png"))
candidate_images = [os.path.relpath(path, "static") for path in candidate_images]
random.seed(42)
random.shuffle(candidate_images)

candidate_images = [
    {"item" : path, "id" : path}
    for path in candidate_images
]

@app.before_request
def get_globals():

    global user, items, queue, tree, current_idx

    username = session.get("user", None)
    if username is None:
        user = None
        items = None
        queue = Queue()
        tree = build_tree(candidate_images[:2], queue=queue)
        current_idx = len(tree)
    else:
        if os.path.isfile(f"{username}.pkl"):
            with open(f"{username}.pkl", "rb") as f:
                data = pickle.load(f)
                user = data["user"]
                items = data["items"]
                queue = data["queue"]
                tree = data["tree"]
                current_idx = data["current_idx"]
        else:
            user = User(username)
            items = None
            queue = Queue()
            tree = build_tree(candidate_images[:2], queue=queue)
            current_idx = len(tree)
            
    gunicorn_logger.log(logging.INFO, "PID: {}".format(os.getpid()))
    gunicorn_logger.log(logging.INFO, f"Before request: {user}")        

    return None

@app.after_request
def save_globals(response):

    global user, items, queue, tree, current_idx, CALLED
    gunicorn_logger.log(logging.INFO, f"After request")
    gunicorn_logger.log(logging.INFO, f"After request: {user}")

    if user is not None:
        data = {
            "user": user,
            "items": items,
            "queue": queue,
            "tree": tree,
            "current_idx": current_idx
        }
        with open(f"{user.name}.pkl", "wb") as f:
            pickle.dump(data, f)

    return response

@app.route("/done")
def done():
    return render_template('done.html')

@app.route("/login", methods=['POST'])
def login():
    global user, tree, queue, current_idx

    user = User(request.form.get('username'))

    if not os.path.isfile("users.json"):
        with open("users.json", "w") as f:
            json.dump({}, f)
    
    with open("users.json", "r") as f:
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
    with open("users.json", "w") as f:
        json.dump(users, f)

    if os.path.exists(f"{user.name}-{CLASS_ID}-tree.pkl"):
        # Clears the previous queue
        queue.clear()

        # Loads the tree from the previous session
        tree = Tree(queue=queue)
        tree.load(f"{user.name}-{CLASS_ID}-tree.pkl")

        current_idx = len(tree)
        if current_idx < len(candidate_images):
            tree = build_tree([candidate_images[current_idx]], tree)
        current_idx += 1

    gunicorn_logger.log(logging.INFO, user)
    gunicorn_logger.log(logging.INFO, "User logged in: {}".format(user.name))

    session['user'] = user.name

    return redirect(url_for('index'))

@app.route('/')
def index():

    global user, items, queue, tree, candidate_images

    gunicorn_logger.log(logging.INFO, "User: {}".format(user))
    gunicorn_logger.log(logging.INFO, "Queue: {}".format(queue))

    if user is None:
        return render_template('login.html')

    # Display the example image and two candidate images
    try:
        items = queue.dequeue()
    except:
        # No more items in the queue to annotate
        items = None
        pass
    
    # Store the items in the global context
    items = items
    gunicorn_logger.log(logging.INFO, (queue.queue, items))

    # When all the images have been annotated, display the done page
    if len(candidate_images) == len(tree) and items is None:
        gunicorn_logger.log(logging.INFO, tree.get_ranking())
        return redirect(url_for('done'))

    if items is None:
        return render_template('index.html', template_image=template_image, candidate_images=[])
    return render_template('index.html', template_image=template_image, candidate_images=[node.get_item() for node in items])

@app.route('/choose', methods=['POST'])
def choose():

    global user, items, queue, tree, current_idx, candidate_images

    # Get the selected image from the form
    selected_image = request.form.get('selected_image')

    gunicorn_logger.log(logging.INFO, "User selected: {}".format(selected_image))
    gunicorn_logger.log(logging.INFO, (queue, items))
    
    # Update the tree with a new choice
    items[0].add_child(items[1], selected_image == items[0].get_item(), queue=queue)
    if len(queue) == 0:
        if current_idx < len(candidate_images):
            tree = build_tree([candidate_images[current_idx]], tree)
        current_idx += 1

        # Save the tree to a file
        tree.save(f"{user.name}-{CLASS_ID}-tree.pkl")

        if current_idx > len(candidate_images):
            return redirect(url_for('index'))

    # Redirect back to the main page after the choice
    return redirect(url_for('index'))

@app.route("/logout", methods=['POST'])
def logout():

    global tree, current_idx, user, queue, candidate_images
    
    with open("users.json", "r") as f:
        users = json.load(f)
        users[user.name] = 0
    with open("users.json", "w") as f:
        json.dump(users, f)

    session.pop('user', None)

    user = None

    # Clears the previous queue and restarts the tree
    queue.clear()
    tree = build_tree(candidate_images[:2], queue=queue)
    current_idx = len(tree)

    return redirect(url_for('index'))

if __name__ == '__main__':
    
    app.run(host="0.0.0.0", port=5000)
