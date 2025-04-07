
import torch
import numpy

from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, "..")
from classification import Query, Template

sys.path.insert(0, "../..")
from model_builder import get_pretrained_model_v2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loads backbone model
model, cfg = get_pretrained_model_v2(
    "mae-lightning-small", 
    weights="MAE_SMALL_STED",
    as_classifier=True,
    num_classes=1,
    blocks="all",
    global_pool="patch",
    mask_ratio=0,
)
model = model.to(DEVICE)

def aggregate_from_templates(templates):
    aggregated = defaultdict(list)
    keys = set()
    for template in templates:
        keys.update(template.keys())
    keys = sorted(list(keys))
    for key in keys:
        for template in templates:
            aggregated[key].extend(template.get(key, []))
    return aggregated

def train_model(clf, data, metadata):

    model_data = {
        "X_train": metadata.get("X_train", None),
        "X_test": metadata.get("X_test", None),
        "y_train": metadata.get("y_train", None),
        "y_test": metadata.get("y_test", None)
    }

    X = numpy.concatenate([values for values in data.values()], axis=0).squeeze()
    y = numpy.concatenate([numpy.ones(len(values)) * i for i, values in enumerate(data.values())], axis=0)

    # Use the previous data to train the model; Only for warm start
    # If the data is already present in the previous data, then do not use it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    if metadata.get("warm_start", False) and model_data["X_train"] is not None:
        distances = cdist(X_train, model_data["X_train"])
        mask = numpy.invert(numpy.any(distances == 0, axis=0))
        if numpy.any(mask):
            X_train = numpy.concatenate([X_train, model_data["X_train"][mask]], axis=0)
            y_train = numpy.concatenate([y_train, model_data["y_train"][mask]], axis=0)

        distances = cdist(X_test, model_data["X_test"])
        mask = numpy.invert(numpy.any(distances == 0, axis=0))
        if numpy.any(mask):
            X_test = numpy.concatenate([X_test, model_data["X_test"][mask]], axis=0)
            y_test = numpy.concatenate([y_test, model_data["y_test"][mask]], axis=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if clf is None:
        clf = RandomForestClassifier(n_estimators=metadata["n_estimators"], random_state=42, max_depth=metadata["max_depth"])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))        

    return clf, {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

class DataHandler:
    def __init__(self, processing_service):
        self.processing_service = processing_service

    def handle_data_request(self, data):
        processed_data = self.processing_service.process_data(data)

        if processed_data.get("test_connection", False):
            return {"status": "ok"}

        if processed_data.get("train", False):
            images = {
                "group0" : {
                    "image" : processed_data["image"],
                    "label" : processed_data["label"]
                }
            }
            template = Template(images, class_id=None, mode="all")
            templates = template.get_template(model, cfg)
            templates = aggregate_from_templates(templates)

            clf = processed_data.get("model", None)
            clf, model_data = train_model(clf, templates, processed_data)     
        else:
            # Assumes the model is already trained
            label = processed_data["image"].copy()
            while label.ndim < 4:
                label = label[numpy.newaxis]
            images = {
                "group0" : {
                    "image" : processed_data["image"],
                    "label" : label
                }
            }
            clf = processed_data.get("model", None)
            model_data = {
                "X_train": processed_data["X_train"],
                "X_test": processed_data["X_test"],
                "y_train": processed_data["y_train"],
                "y_test": processed_data["y_test"]
            }
        
        query = Query(images, class_id=None)
        result = [query_result for query_result in query.query(model, clf, cfg)][0]
        result["prediction"] += 1
        result["prediction"] = result["prediction"][0]
        result["model"] = clf
        result.update(**model_data)

        transformed_result = self.processing_service.transform_data(result)

        # Remove image and label from the result
        transformed_result.pop("image")
        transformed_result.pop("label")

        return transformed_result

    def handle_batch_request(self, data_list):
        results = []
        for data in data_list:
            result = self.handle_data_request(data)
            results.append(result)
        return results