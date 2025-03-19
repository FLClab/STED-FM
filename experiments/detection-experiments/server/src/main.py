import json
import torch

from flask import Flask, request, jsonify
from handlers.data_handler import DataHandler
from services.processing_service import ProcessingService

app = Flask(__name__)

processing_service = ProcessingService()
data_handler = DataHandler(processing_service)

@app.route('/process', methods=['POST'])
def process_data():

    incoming_data = request.data.decode("utf-8")
    incoming_data = json.loads(incoming_data)

    result = data_handler.handle_data_request(incoming_data)

    return jsonify(result)

if __name__ == '__main__':

    app.run(host="0.0.0.0", port="5090", debug=True)