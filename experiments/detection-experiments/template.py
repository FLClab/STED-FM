
import numpy
import torch
import random
import tifffile

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

        self.mean, self.std = numpy.mean(self.x, axis=(-2, -1), keepdims=True), numpy.std(self.x, axis=(-2, -1), keepdims=True)

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
            # mean, std = numpy.array([mean, mean, mean]), numpy.array([std, std, std])
            # mean, std = numpy.array([0.014, 0.014, 0.014]), numpy.array([0.03, 0.03, 0.03])
            image_crop = (image_crop - self.mean[image_idx]) / self.std[image_idx]       

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
    
    def _get_all_templates_vit(self, model, cfg):

        def compute_template(image_crop, mask_crop):
            image_crop = image_crop[numpy.newaxis, ...]
            if cfg.in_channels == 3:
                mean, std = numpy.mean(image_crop), numpy.std(image_crop)
                image_crop = numpy.repeat(image_crop, 3, axis=0)
                mean, std = numpy.array([mean, mean, mean]), numpy.array([std, std, std])
                # mean, std = numpy.array([0.014, 0.014, 0.014]), numpy.array([0.03, 0.03, 0.03])
                image_crop = (image_crop - mean[:, numpy.newaxis, numpy.newaxis]) / std[:, numpy.newaxis, numpy.newaxis]
            image_crop = image_crop.astype(numpy.float32)

            # fig, axes = pyplot.subplots(1, 2)
            # axes[0].imshow(image_crop[0], cmap="gray", vmin=0, vmax=0.7*image_crop.max())
            # axes[1].imshow(mask_crop)
            # fig.savefig("crop.png")
            # print(len(templates))
            # input("Press Enter to continue...")
            # pyplot.close()

            image_crop = torch.tensor(image_crop).unsqueeze(0).to(next(model.parameters()).device)
            features = model.forward_features(image_crop)
            features = features.cpu().squeeze().numpy()

            m = measure.block_reduce(mask_crop, (16, 16), numpy.mean)
            threshold = 0.5 * m.max()

            patch_idx = numpy.argmax(m.ravel())
            template = features[patch_idx]

            patch_indices = numpy.argwhere(m.ravel() > threshold)
            template = features[patch_indices]
            template = numpy.mean(template, axis=0).ravel()

            image_crop = image_crop.cpu().numpy()[0, 0]

            return {
                "template" : template,
                "image-template" : image_crop,
                "mask-template" : mask_crop
            }           

        templates = []
        for key, values in self.images.items():

            images = values["image"]
            # if cfg.in_channels != 3:
            m, M = numpy.min(images, axis=(-2, -1), keepdims=True), numpy.max(images, axis=(-2, -1), keepdims=True)
            # M = numpy.quantile(images, 0.995, axis=(-2, -1), keepdims=True)
            images = (images - m) / (M - m + 1e-6)
            images = numpy.clip(images, 0, 1)

            labels = values["label"]

            for image, label in zip(images, labels):

                label = measure.label((label[self.class_id] > 0).astype(int))

                if image.shape[0] == 224 and image.shape[1] == 224:
                    image_crop = image
                    mask_crop = label

                    template = compute_template(image_crop, mask_crop)
                    templates.append(template)
                else:
                    rprops = measure.regionprops(label)
                    for rprop in rprops:
                        r, c = rprop.centroid
                        r, c = int(r), int(c)

                        image_crop = image[
                            max(0, r - self.size // 2) : min(r + self.size // 2, image.shape[0]),
                            max(0, c - self.size // 2) : min(c + self.size // 2, image.shape[1]),
                        ]
                        image_crop = numpy.pad(
                            image_crop,
                            (
                                (max(0, self.size // 2 - r), max(0, r + self.size // 2 - image.shape[0])),
                                (max(0, self.size // 2 - c), max(0, c + self.size // 2 - image.shape[1])),
                            ),
                        )
                        mask_crop = label[
                            max(0, r - self.size // 2) : min(r + self.size // 2, image.shape[0]),
                            max(0, c - self.size // 2) : min(c + self.size // 2, image.shape[1]),
                        ]
                        mask_crop = numpy.pad(
                            mask_crop,
                            (
                                (max(0, self.size // 2 - r), max(0, r + self.size // 2 - image.shape[0])),
                                (max(0, self.size // 2 - c), max(0, c + self.size // 2 - image.shape[1])),
                            ),
                        )

                        template = compute_template(image_crop, mask_crop)
                        templates.append(template)
        return templates
    
    def _get_stack_template_vit(self, model, cfg):
        templates = self._get_all_templates_vit(model, cfg)
        return numpy.stack([template["template"] for template in templates], axis=0)
    
    def _get_avg_template_vit(self, model, cfg):
        templates = self._get_all_templates_vit(model, cfg)
        templates = [template["template"] for template in templates]
        return numpy.mean(templates, axis=0)
    
    def _get_random_template_vit(self, model, cfg):
        templates = self._get_all_templates_vit(model, cfg)
        return random.choice(templates)["template"]
        # return numpy.mean(templates, axis=0)    

    def _get_choice_template_vit(self, model, cfg, choice=None):
        if choice is None:
            if self.class_id == 1:
                choice = 224
            elif self.class_id == 2:
                choice = 5
        templates = self._get_all_templates_vit(model, cfg)
        return templates[choice]
    
    def _get_choices_template_vit(self, model, cfg, choices=None):
        """
        CHANID = 0 -- Perforated -- 5, 7, 9, 12, 16, 19, 30, 33, 34, 35, 36, 38, 43, 44, 45, 57, 60
        CHANID = 0 -- Elongated -- 19, 42, 47, 90, 99, 147, 148, 173, 185, 186, 224, 225
        """
        if choices is None:
            if self.class_id == 1:
                choices = [19, 42, 47, 90, 99, 147, 148, 173, 185, 186, 224, 225]
            elif self.class_id == 2:
                choices = [5, 7, 9, 12, 16, 19, 30, 33, 34, 35, 36, 38, 43, 44, 45, 57, 60]

        templates = self._get_all_templates_vit(model, cfg)
        return {
            "image-template" : numpy.stack([templates[choice]["image-template"] for choice in choices], axis=0),
            "mask-template" : numpy.stack([templates[choice]["mask-template"] for choice in choices], axis=0),
            "template" : numpy.mean([templates[choice]["template"] for choice in choices], axis=0),
        }
        # return numpy.mean([templates[choice]["template"] for choice in choices], axis=0)
    
    def _get_avg_template_convnet(self, model):
            raise NotImplementedError("Not yet implemented")


class Query:
    def __init__(self, images, class_id, size=224):
        self.images = images
        self.class_id = class_id
        self.size = size
    
    def query(self, template, model, cfg):
        model.eval()
        if "vit" in cfg.backbone:
            model_type = "vit"
        else:
            model_type = "convnet"
        return getattr(self, f"_query_image_{model_type}")(template, model, cfg)   
    
    def _query_image_vit(self, template, model, cfg):

        template = torch.tensor(template)
        if template.ndim == 1:
            template = template.unsqueeze(0).unsqueeze(0)
        elif template.ndim == 2:
            template = template.unsqueeze(1)
        template = template.to(next(model.parameters()).device)

        for key, values in self.images.items():
            
            if isinstance(values, dict):
                images = values["image"]
                labels = values["label"]
            else:
                images = values
                labels = values

            for image, label in zip(tqdm(images, desc=f"Images ({key})", leave=False), labels):
                
                image_name = None
                if isinstance(image, str):
                    image_name = image
                    image = tifffile.imread(image)
                    if image.ndim == 3:
                        image = image[0]
                    # This assumes that there is no label
                    label = image.copy()[numpy.newaxis]

                # if cfg.in_channels != 3:
                m, M = numpy.min(image, axis=(-2, -1), keepdims=True), numpy.max(image, axis=(-2, -1), keepdims=True)
                image = (image - m) / (M - m + 1e-6)
                image = numpy.clip(image, 0, 1)

                dataset = ImageDataset(image, label, in_channels=cfg.in_channels, size=self.size, step=int(self.size * 0.25))
                sampler = OnTheFlySampler(dataset)
                loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
                builder = PredictionBuilder(image.shape, self.size, num_classes=1)
                for X, y, positions in loader:

                    if X.ndim == 3:
                        X = X.unsqueeze(1)
                    X = X.to(next(model.parameters()).device)
                    features = model.forward_features(X).unsqueeze(1)
                    distances = torch.nn.functional.cosine_similarity(features, template, dim=3)
                    distances = distances.cpu().detach().numpy()

                    # Max-pooling
                    distances = numpy.max(distances, axis=1)

                    distances = distances.reshape(-1, 14, 14)
                    distances = rescale(distances, (1, 16, 16), order=0, anti_aliasing=False)
                    
                    for pred, j, i in zip(distances, *positions):
                        builder.add_predictions_ji(pred, j, i)
                prediction = builder.return_prediction()
                
                yield {"image" : image,
                        "label" : label,
                        "prediction" : prediction[0],
                        "image-name" : image_name,
                        "condition" : key}

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