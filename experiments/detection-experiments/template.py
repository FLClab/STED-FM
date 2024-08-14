
import numpy
import torch

from skimage import measure

class Template:
    def __init__(self, images, class_id, mode="avg", size=224):
        self.images = images
        self.class_id = class_id
        self.mode = mode
        self.size = size

    def get_template(self, model, cfg):
        if "vit" in cfg.backbone:
            model_type = "vit"
        else:
            model_type = "convnet"
        return getattr(self, f"_get_{self.mode}_template_{model_type}")(model)

    def _get_avg_template_vit(self, model):
        
        templates = []
        for key, values in self.images.items():
            images = values["image"]
            m, M = numpy.min(images, axis=(-2, -1), keepdims=True), numpy.max(images, axis=(-2, -1), keepdims=True)
            images = (images - m) / (M - m + 1e-6)

            labels = values["label"]

            for image, label in zip(images, labels):
                label = measure.label((label[self.class_id] > 0).astype(int))
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

                    image_crop = image_crop.astype(numpy.float32)
                    image_crop = torch.tensor(image_crop).unsqueeze(0).unsqueeze(0).to(next(model.parameters()).device)
                    features = model.forward_features(image_crop)
                    features = features.cpu().squeeze().numpy()

                    m = measure.block_reduce(mask_crop, (16, 16), numpy.mean)
                    patch_idx = numpy.argmax(m.ravel())
                    template = features[patch_idx]
                    templates.append(template)

        return numpy.mean(templates, axis=0)
    
def _get_avg_template_convnet(self, model):
        raise NotImplementedError("Not yet implemented")



                    