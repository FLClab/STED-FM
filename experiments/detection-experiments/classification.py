
import numpy
import torch
import random
import tifffile
import os

from matplotlib import pyplot
from skimage import measure, feature
from skimage.transform import rescale
from torch.utils.data import Sampler, Dataset, DataLoader
from scipy.spatial import distance
from tqdm import tqdm

CHANID = 0

class ImageDataset(Dataset):
    def __init__(self, image, label, in_channels=1, size=224, step=224, **kwargs):
        super().__init__()
        self.x = image
        self.y = label

        if self.x.ndim == 2:
            self.x = self.x[numpy.newaxis, ...]
        if self.y.ndim == 3:
            self.y = self.y[numpy.newaxis, ...]

        self.size = size
        self.step = step
        self.in_channels = in_channels

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        image_idx, (j, i) = index
        image_crop = self.x[image_idx, j - self.size // 2 : j + self.size // 2, i - self.size // 2 : i + self.size // 2]
        label_crop = self.y[image_idx, :, j - self.size // 2 : j + self.size // 2, i - self.size // 2 : i + self.size // 2]

        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "constant")
            label_crop = numpy.pad(label_crop, ((0, 0), (0, self.size - label_crop.shape[1]), (0, self.size - label_crop.shape[2])), "constant")
        
        image_crop = image_crop.astype(numpy.float32)
        label_crop = label_crop.astype(numpy.float32)

        image_crop = image_crop[numpy.newaxis, ...]
        if self.in_channels == 3:
            image_crop = numpy.repeat(image_crop, 3, axis=0)
            mean = numpy.mean(image_crop, axis=(-2, -1))
            std = numpy.std(image_crop, axis=(-2, -1))
            mean, std = numpy.array([mean, mean, mean]), numpy.array([std, std, std])
            mean, std = numpy.array([0.014, 0.014, 0.014]), numpy.array([0.03, 0.03, 0.03])
            image_crop = (image_crop - mean[:, numpy.newaxis, numpy.newaxis]) / std[:, numpy.newaxis, numpy.newaxis]            

        image_crop = torch.tensor(image_crop, dtype=torch.float32)
        label_crop = torch.tensor(label_crop)
        return image_crop, label_crop, (j, i)

class OnTheFlySampler(Sampler):
    """
    Creates a `Sampler` that samples all crops in an image instance in sliding
    window fashion.
    To use this `Sampler` the `Dataet` instance should have the {shape} key

    :param dataset: A `Dataset` instance
    """
    def __init__(self, dataset, **kwargs):
        super().__init__()
        self.dataset = dataset

        self.samples_per_image = self.generate_samples()

    def generate_samples(self):
        """
        Generates all samples in an image based on the shape of the image
        """
        samples_per_image = []
        for i in range(len(self.dataset.x)):
            samples = []
            shape = self.dataset.x.shape[-2:]
            for j in range(0, shape[0], self.dataset.step):
                for i in range(0, shape[1], self.dataset.step):
                    samples.append((j + self.dataset.size // 2, i + self.dataset.size // 2))
            samples_per_image.append(samples)
        return samples_per_image

    def __iter__(self):
        """
        Creates the iterator of the `Sampler`
        """
        samples = []
        for i in range(len(self.dataset)):
            for sample in self.samples_per_image[i]:
                samples.append((i, sample))
        return iter(samples)

    def __len__(self):
        """
        Returns the total number of samples
        """
        return sum([len(samples) for samples in self.samples_per_image])

class PredictionBuilder:
    """
    This class is used to create the final prediction from the predictions
    that are infered by the network. This class stores the predictions in an output
    array to avoid memory overflow with the method `add_predictions` and then
    computes the mean prediction of the overlap with the `return_prediction` method.

    :param shape: The shape of the image
    :param size: The size of the crops
    """
    def __init__(self, shape, size, num_classes=2):
        # Assign member variables
        self.shape = shape
        self.size = size

        # Creates the output arrays
        self.pred = numpy.zeros((num_classes, self.shape[0] + self.size, self.shape[1] + self.size), dtype=numpy.float32)
        self.pixels = numpy.zeros((self.shape[0] + self.size, self.shape[1] + self.size), dtype=numpy.float32)

    def add_predictions(self, predictions, positions):
        """
        Method to store the predictions in the output arrays. We suppose positions
        to be central on crops

        :param predictions: A `numpy.ndarray` of predictions with size (batch_size, features, H, W)
        :param positions: A `numpy.ndarray` of positions of crops with size (batch_size, 2)
        """
        # Verifies the shape of predictions
        if predictions.ndim != 4:
            # The feature channel has a high probabilty of missing
            predictions = predictions[:, numpy.newaxis, ...]
        for pred, (j, i) in zip(predictions, positions):

            # Stores the predictions in output arrays
            self.pred[:, j - self.size // 2 : j + self.size // 2, i - self.size // 2 : i + self.size // 2] += pred
            self.pixels[j - self.size // 2 : j + self.size // 2, i - self.size // 2 : i + self.size // 2] += 1

    def add_predictions_ji(self, prediction, j, i):
        """
        Method to store the predictions in the output array at the corresponding
        position. We suppose a central postion of the crop

        :param predictions: A `numpy.ndarray` of prediction with size (features, H, W)
        :param j: An `int` of the row position
        :param i: An `int` of the column position
        """
        # Verifies the shape of prediction
        if prediction.ndim != 3:
            prediction = prediction[numpy.newaxis, ...]

        # Crops image if necessary
        slc = (
            slice(None, None),
            slice(
                0 if j - self.size // 2 >= 0 else -1 * (j - self.size // 2),
                prediction.shape[-2] if j + self.size // 2 < self.pred.shape[-2] else self.pred.shape[-2] - (j + self.size // 2)
            ),
            slice(
                0 if i - self.size // 2 >= 0 else -1 * (i - self.size // 2),
                prediction.shape[-1] if i + self.size // 2 < self.pred.shape[-1] else self.pred.shape[-1] - (i + self.size // 2)
            )
        )
        pred = prediction[slc]

        # Stores prediction in output arrays
        self.pred[:, max(0, j - self.size // 2) : j + self.size // 2,
                     max(0, i - self.size // 2) : i + self.size // 2] += pred
        self.pixels[max(0, j - self.size // 2) : j + self.size // 2,
                    max(0, i - self.size // 2) : i + self.size // 2] += 1

    def return_prediction(self):
        """
        Method to return the final prediction.

        :returns : The average prediction map from the overlapping predictions
        """
        self.pixels[self.pixels == 0] += 1 # Avoids division by 0
        return (self.pred / self.pixels)[:, :self.shape[0], :self.shape[1]]

class Template:
    def __init__(self, images, class_id, mode="avg", size=224):
        self.images = images
        self.class_id = class_id
        self.mode = mode
        self.size = size

    def get_template(self, model, cfg):
        model.eval()
        if "vit" in cfg.backbone:
            model_type = "vit"
        else:
            model_type = "convnet"
        return getattr(self, f"_get_{self.mode}_template_{model_type}")(model, cfg)
    
    def _get_all_template_vit(self, model, cfg):

        def compute_template(mask_crop, features):
            templates_per_channels = {}
            for channel in range(mask_crop.shape[0]):
                if not numpy.any(mask_crop[channel]):
                    continue

                m = measure.block_reduce(mask_crop[channel], (16, 16), numpy.mean)
                threshold = 0.5 * m.max()

                patch_indices = numpy.argwhere(m.ravel() > threshold)
                template = features[0, patch_indices]
                templates_per_channels[channel] = template
            return templates_per_channels
        
        templates = []
        for key, values in self.images.items():
            
            if isinstance(values, dict):
                images = values["image"]
                labels = values["label"]
            else:
                images, labels = [], []
                for image_name in values:
                    label_name = image_name.replace(".tif", "_annotations.tif")
                    if os.path.isfile(label_name):
                        images.append(image_name)
                        labels.append(label_name)

            for image, label in zip(tqdm(images, desc=f"Images ({key})", leave=False), labels):
                
                image_name, label_name = None, None
                if isinstance(image, str):
                    image_name = image
                    image = tifffile.imread(image)
                    if image.ndim == 3:
                        image = image[0]
                    
                    label_name = label
                    label = tifffile.imread(label)

                # if cfg.in_channels != 3:
                m, M = numpy.min(image, axis=(-2, -1), keepdims=True), numpy.max(image, axis=(-2, -1), keepdims=True)
                image = (image - m) / (M - m + 1e-6)
                image = numpy.clip(image, 0, 1)

                dataset = ImageDataset(image, label, in_channels=cfg.in_channels, size=self.size, step=int(self.size * 0.5))
                sampler = OnTheFlySampler(dataset)
                loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
                builder = PredictionBuilder(image.shape, self.size, num_classes=1)
                for X, y, positions in loader:

                    if X.ndim == 3:
                        X = X.unsqueeze(1)
                    X = X.to(next(model.parameters()).device)
                    features = model.forward_features(X).unsqueeze(1)

                    y = y.cpu().detach().numpy()
                    features = features.cpu().detach().numpy()

                    for image_crop, label_crop, feature in zip(X, y, features):
                        if numpy.any(label_crop):
                            template = compute_template(label_crop, feature)
                            templates.append(template)
        return templates    
    
    def _get_avg_template_convnet(self, model):
            raise NotImplementedError("Not yet implemented")


class Query:
    def __init__(self, images, class_id, size=224):
        self.images = images
        self.class_id = class_id
        self.size = size
    
    def query(self, model, clf, cfg):
        model.eval()
        if "vit" in cfg.backbone:
            model_type = "vit"
        else:
            model_type = "convnet"
        return getattr(self, f"_query_image_{model_type}")(model, clf, cfg)   
    
    def _query_image_vit(self, model, clf, cfg):

        for key, values in self.images.items():
            
            if isinstance(values, dict):
                images = values["image"]
                labels = values["label"]
            else:
                images = values
                labels = [None] * len(images)

            for image, label in zip(tqdm(images, desc=f"Images ({key})", leave=False), labels):
                
                image_name = None
                if isinstance(image, str):
                    image_name = image
                    image = tifffile.imread(image)
                    if image.ndim == 3:
                        image = image[0]
                    label = image.copy()[numpy.newaxis]

                # if cfg.in_channels != 3:
                m, M = numpy.min(image, axis=(-2, -1), keepdims=True), numpy.max(image, axis=(-2, -1), keepdims=True)
                image = (image - m) / (M - m + 1e-6)
                image = numpy.clip(image, 0, 1)

                dataset = ImageDataset(image, label, in_channels=cfg.in_channels, size=self.size, step=int(self.size * 1.0))
                sampler = OnTheFlySampler(dataset)
                loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
                builder = PredictionBuilder(image.shape, self.size, num_classes=1)
                for X, y, positions in loader:

                    if X.ndim == 3:
                        X = X.unsqueeze(1)
                    X = X.to(next(model.parameters()).device)
                    features = model.forward_features(X).unsqueeze(1)
                    features = features.cpu().detach().numpy()
                    
                    batch_size = features.shape[0]
                    features = numpy.reshape(features, (-1, features.shape[-1]))
                    y_pred = clf.predict(features)[..., numpy.newaxis]

                    reshaped = numpy.reshape(y_pred, (batch_size, 14, 14, -1))
                    reshaped = numpy.transpose(reshaped, (0, 3, 1, 2))
                    reshaped = rescale(reshaped, (1, 1, 16, 16), anti_aliasing=False, order=0)

                    for p, j, i in zip(reshaped, *positions):
                        builder.add_predictions_ji(p, j, i)

                prediction = builder.return_prediction()
                yield {
                    "image" : image,
                    "label" : label,
                    "prediction" : prediction,
                    "image-name" : image_name,
                    "condition" : key
                }

    def _query_image_convnet(self, template, model, cfg):
        raise NotImplementedError("Not yet implemented")
    
def sample_topk(image, prediction, shape=(80, 80), k=5):
    """
    Sample the top-k predictions from the prediction map

    :param image: The input image to sample from
    :param prediction: The prediction map
    :param shape: The shape of the crop
    :param k: The number of crops to sample

    :returns : A list of crops and their corresponding coordinates
    """
    if isinstance(shape, (int, float)):
        shape = (shape, shape)

    coords = feature.peak_local_max(prediction, min_distance=shape[0]//2, num_peaks=k, exclude_border=shape)
    crops = []
    for coord in coords:
        j, i = coord
        crop = image[j - shape[0]//2 : j + shape[0]//2, i - shape[1]//2 : i + shape[1]//2]
        crops.append(crop)

    intensity = prediction[coords[:, 0], coords[:, 1]]
    argsort = numpy.argsort(intensity)[::-1]

    return [crops[arg] for arg in argsort], [coords[arg] for arg in argsort]