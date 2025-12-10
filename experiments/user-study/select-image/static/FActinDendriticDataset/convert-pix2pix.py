
import glob
import os
import numpy 
import tifffile
from PIL import Image
from PIL.Image import Resampling
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm

images = glob.glob(os.path.join("./raw-pix2pix", "*.png"))
output_dir = "./candidates"

def optimize_intensity(image, template):
    def loss_func(params):
        scale, shift = params
        adjusted = image * scale + shift
        return numpy.mean((adjusted - template) ** 2)

    initial_params = [1.0, 0.0]
    result = minimize(loss_func, initial_params, method='L-BFGS-B')
    optimized_scale, optimized_shift = result.x
    optimized_image = image * optimized_scale + optimized_shift
    optimized_image = numpy.clip(optimized_image, 0, 1)
    return optimized_image

for image_path in tqdm(images):

    template_image = os.path.basename(image_path)
    idx = template_image.split("_")[2].split(".")[0]
    template = os.path.join("./raw", f"sample_{idx}_sted.tif")
    template_image = tifffile.imread(template)

    image = Image.open(image_path)

    # Resize image to 224x224
    image = image.resize((224, 224), Resampling.BILINEAR)
    image = numpy.array(image).astype(numpy.float32)
    image = image[..., 0]  # Use only one channel/grayscale

    image = (image - image.min()) / (image.max() - image.min())

    # Export to tiff
    tifffile.imwrite(os.path.join("./raw", f"sample_{idx}_pix2pix.tif"), image.astype(numpy.float32))

    # optimized_image = optimize_intensity(image, template_image)

    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, base_name)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="hot")
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=900)
    plt.close(fig)

    # image.save(output_path)