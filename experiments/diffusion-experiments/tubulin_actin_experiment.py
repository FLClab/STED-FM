import numpy as np 
import matplotlib.pyplot as plt
import argparse 
import torch 
from torch import nn 
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet 
from tqdm import tqdm 
import sys 
from attribute_datasets import TubulinActinDataset 
import glob 
from stedfm import get_pretrained_model_v2 
from stedfm.DEFAULTS import BASE_PATH, COLORS 
from stedfm.utils import set_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--boundary", type=str, default="tubulin-actin")
parser.add_argument("--num-samples", type=int, default=10)
parser.add_argument("--ckpt-path", type=str, default="/home-local/Frederic/baselines/DiffusionModels/latent-guidance")
parser.add_argument("--figure", action="store_true")
parser.add_argument("--sanity-check", action="store_true")
parser.add_argument("--direction", type=str, default="actin")
args = parser.parse_args()

def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-4.0,
                       end_distance=4.0,
                       steps=8):
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary, linspace
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1), linspace
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] but {latent_code.shape} was received.')

def compute_power_spectrum(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    ft = np.fft.fft2(image)
    ft_shift = np.fft.fftshift(ft)
    mag = np.log(np.abs(ft_shift))
    y, x = np.indices(mag.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = mag.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr
    freq = np.arange(radial_prof.shape[0]) / (224 * 20)
    return freq, radial_prof


def load_boundary() -> np.ndarray:
    print(f"--- Loading boundary trained from {args.weights} embeddings ---")
    data = np.load(f"./tubulin-actin-experiment/boundaries/{args.weights}_{args.boundary}_boundary.npz")
    boundary, intercept, norm = data["boundary"], data["intercept"], data["norm"]
    return boundary, intercept, norm

def save_power_spectra(power_spectra, index, freq):
    keys = ["original", "sample", "lerp1", "lerp2", "lerp3", "lerp4"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mask = np.logical_and(freq <= 0.0059, freq >= 0.005)
    for i, (k, p) in enumerate(zip(keys, power_spectra)):
        ax.plot(freq[mask], p[mask] / p[0], label=k)
    ax.legend()
    fig.savefig(f"./tubulin-actin-experiment/examples/{args.weights}-powerspectrum_{index}.pdf", dpi=1200, bbox_inches="tight")
    plt.close()

def save_examples(samples, distances, index):
   N = len(samples)
   fig, axs = plt.subplots(1, N, figsize=(10,5))
   for i, (s, d) in enumerate(zip(samples, distances)):
       if s.shape[0] == 3:
           s = s[0]
       axs[i].imshow(s, cmap="hot", vmin=0.0, vmax=1.0)
       axs[i].set_title(f"Distance: {d:.2f}")
       axs[i].axis("off")
   plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
   fig.savefig(f"./tubulin-actin-experiment/examples/{args.weights}-image_{index}_to{args.direction}.pdf", dpi=1200, bbox_inches="tight")
   plt.close(fig)


def plot_distance_distribution(distances_to_boundary: dict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(distances_to_boundary["tubulin"], bins=100, alpha=0.5, label="Tubulin")
    ax.hist(distances_to_boundary["actin"], bins=100, alpha=0.5, label="Actin")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"./tubulin-actin-experiment/examples/sanity-check/{args.weights}-distance_distribution.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def main():
    if args.figure:
        pass 
    elif args.sanity_check:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary, intercept, norm = load_boundary() 
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.latent_encoder,
            weights=args.weights,
            path=None,
            mask_ratio=0.0,
            pretrained=False,
            in_channels=1,
            as_classifier=True,
            blocks="all",
            num_classes=4
        )
        latent_encoder.to(DEVICE)
        latent_encoder.eval()
        dataset = TubulinActinDataset(
        data_folder="/home-local/Frederic/evaluation-data/optim_train",
        classes=["tubulin", "actin"],
        n_channels=1,
        min_quality_score=0.70
        )
        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)

        distances_to_boundary = {"tubulin": [], "actin": []}
        with torch.no_grad():
            for i in tqdm(indices, total=N):
                img, metadata = dataset[i]
                score = metadata["score"]
                assert score >= 0.70 
                label = metadata["label"]
                torch_img = img.clone().detach().unsqueeze(0).to(DEVICE)
                latent_code = latent_encoder.forward_features(torch_img)
                numpy_code = latent_code.detach().cpu().numpy()
                d = numpy_code.dot(boundary.T) + intercept
                d = d[0][0]
                key = "tubulin" if label == 0 else "actin"
                distances_to_boundary[key].append(d)
        plot_distance_distribution(distances_to_boundary)



    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary, intercept, norm = load_boundary()
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.latent_encoder,
            weights=args.weights,
            path=None, 
            mask_ratio=0.0,
            pretrained=False,
            in_channels=3 if "imagenet" in args.weights.lower() else 1,
            as_classifier=True,
            blocks="all",
            num_classes=4
        )
        denoising_model = UNet(
            dim=64,
            channels=1, 
            dim_mults=(1,2,4),
            cond_dim=model_config.dim,
            condition_type="latent",
            num_classes=4
        )
        diffusion_model = DDPM(
            denoising_model=denoising_model,
            timesteps=args.timesteps,
            beta_schedule="linear",
            condition_type="latent",
            latent_encoder=latent_encoder,
        )

        ckpt_path = torch.load(f"{args.ckpt_path}/{args.weights}/checkpoint-69.pth")
        diffusion_model.load_state_dict(ckpt_path["state_dict"])
        diffusion_model.to(DEVICE)

        dataset = TubulinActinDataset(
            data_folder="/home-local/Frederic/evaluation-data/optim-data",
            classes=["tubulin", "actin"],
            n_channels=1,
            min_quality_score=0.70
        )
        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)
        counter = 0
        n_steps = 4 
        # all_power_spectra = np.zeros((args.num_samples, 156, n_steps+2))
        # all_distances = np.zeros((args.num_samples, n_steps+2))
        num_anchors = 0
        for i in tqdm(indices):
            if counter >= args.num_samples:
                break
            power_spectra, distances = [], [] 
            img, metadata = dataset[i]
            score = metadata["score"]
            assert score >= 0.70 
            target_label = 0 if args.direction == "actin" else 1
            multiplier = 1 if args.direction == "actin" else -1
            if args.boundary == "tubulin-actin" and metadata["label"] != target_label:
                continue 
            else:
                num_anchors += 1
                continue

            if "imagenet" in args.weights.lower():
                img = torch.tensor(img, dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
            else:
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            original = img.squeeze().detach().cpu().numpy()
            # original_freq, original_power_spectrum = compute_power_spectrum(original)
            
            # Ensures reproducibility
            seed_offset = hash(args.direction) % (2**32-1)
            set_seeds(args.seed + i + seed_offset)            

            latent_code = diffusion_model.latent_encoder.forward_features(img)
            numpy_code = latent_code.detach().cpu().numpy()
            original_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=latent_code, progress=True)
            # _, original_sample_power_spectrum = compute_power_spectrum(original_sample.squeeze().detach().cpu().numpy())

            samples = [original, original_sample.squeeze().detach().cpu().numpy()]
            #power_spectra.append(original_power_spectrum)
            #power_spectra.append(original_sample_power_spectrum)
            distances.extend([0.0, 0.0])

            lerped_codes, d = linear_interpolate(latent_code=numpy_code, boundary=boundary, start_distance=multiplier*0.1, end_distance=multiplier*1.5, steps=n_steps)

            for c, code in enumerate(lerped_codes):
                lerped_code = torch.tensor(code, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lerped_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=lerped_code, progress=True)
                lerped_sample_numpy = lerped_sample.squeeze().detach().cpu().numpy()
                samples.append(lerped_sample_numpy)
                #_, lerped_sample_power_spectrum = compute_power_spectrum(lerped_sample_numpy)
                
                #power_spectra.append(lerped_sample_power_spectrum)
                distances.append(abs(d[c][0]))

            save_examples(samples, distances, counter)
            # save_power_spectra(power_spectra, counter, freq=original_freq)
            counter += 1
        print(f"Number of anchors: {num_anchors}")
        



if __name__ == "__main__":
    main()