
import numpy
import torch
import pickle
import os
import json
import h5py 

from torch import nn

from dlutils.utils import load_ckpt

def other_load_ckpt(output_folder, network_name=None, model="UNet", filename="checkpoints.hdf5",
                verbose=True, epoch="best"):
    """
    Saves the current network state to a hdf5 file. The architecture of the hdf5
    file is
    hdf5file
        MICRANet
            network

    :param output_folder: A `str` to the output folder
    :param networks: A `dict` of network models
    :param filename: (optional) A `str` of the filename. Defaults to "checkpoints.hdf5"
    :param verbose: (optional) Wheter the function in verbose
    """
    if verbose:
        print("[----]     Loading network state")
    with h5py.File(os.path.join(output_folder, filename), "r") as file:
        if model in file:
            model_group = file[model]
            if not isinstance(network_name, str):
                network_name = list(model_group.keys())[0]
            state_dict = {k : torch.tensor(v[()]) for k, v in model_group[network_name].items()}
        else:
            if epoch == "max":
                epoch = str(sorted([int(key) for key in file.keys() if key != "best" ])[-1])
            main_group = file[epoch]

            networks = {}
            for key, values in main_group["network"].items():
                networks[key] = {k : torch.tensor(v[()]) for k, v in values.items()}
            state_dict = networks[list(networks.keys())[0]]
    return state_dict

class DoubleConvolver(nn.Module):
    """
    Class for the double convolution in the contracting path. The kernel size is
    set to 3x3 and a padding of 1 is enforced to avoid lost of pixels. The convolution
    is followed by a batch normalization and relu.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Contracter(nn.Module):
    """
    Class for the contraction path. Max pooling of the input tensor is
    followed by the double convolution.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Contracter, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvolver(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Expander(nn.Module):
    """
    Class for the expansion path. Upsampling with a kernel size of 2 and stride 2
    is performed and followed by a double convolution following the concatenation
    of the skipping link information from higher layers.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Expander, self).__init__()
        self.expand = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolver(in_channels=in_channels, out_channels=out_channels)

    def center_crop(self, links, target_size):
        _, _, links_height, links_width = links.size()
        diff_x = (links_height - target_size[0]) // 2
        diff_y = (links_width - target_size[1]) // 2
        return links[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        x = self.expand(x)
        crop = self.center_crop(bridge, x.size()[2 : ])
        concat = torch.cat([x, crop], 1)
        x = self.conv(concat)
        return x


class UNet(nn.Module):
    """
    Class for creating the UNet architecture. A first double convolution is performed
    on the input tensor then the contracting path is created with a given depth and
    a preset number of filters. The number of filter is doubled at every step.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of output channels (i.e. number of classes)
    :param number_filter: Number of filters in the first layer (2 ** number_filter)
    :param depth: Depth of the network
    :param size: The size of the crops that are fed to the network
    """
    def __init__(self, in_channels, out_channels, number_filter=4, depth=4, size=128, *args, **kwargs):
        super(UNet, self).__init__()
        self.size = size
        self.out_channels = out_channels

        self.input_conv = DoubleConvolver(in_channels=in_channels, out_channels=2**number_filter)

        self.contracting_path = nn.ModuleList()
        for i in range(depth - 1):
            self.contracting_path.append(
                Contracter(in_channels=2**(number_filter + i), out_channels=2**(number_filter + i + 1))
            )

        self.expanding_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.expanding_path.append(
                Expander(in_channels=2**(number_filter + i + 1), out_channels=2**(number_filter + i))
            )

        self.output_conv = nn.Conv2d(in_channels=2**number_filter, out_channels=out_channels, kernel_size=1)

        pretrained = kwargs.get("pretrained", None)
        if pretrained:
            self.load_pretrained(pretrained)

    def forward(self, x):
        links = [] # keeps track of the links
        x = self.input_conv(x)
        links.append(x)

        # Contracting path
        for i, contracting in enumerate(self.contracting_path):
            x = contracting(x)
            if i != len(self.contracting_path) - 1:
                links.append(x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # # Expanding path
        # for i, expanding in enumerate(self.expanding_path):
        #     x = expanding(x, links[- i - 1])
        # x = self.output_conv(x)

        # # Sigmoid layer
        # x = torch.sigmoid(x)
        return x

    def load_pretrained(self, path, location="cpu", load_model="best", **kwargs):
        """
        Loads a pretrained model from path

        :param path: A `str` of the path of the model
        """
        print(f"[%%%%] Loading pretrained model from: {path}")
        model_path = os.path.join(path, "params.net")
        if os.path.isfile(model_path):
            net_params = torch.load(model_path,
                                    map_location=location)
        else:
            try:
                net_params, _, _, _ = load_ckpt(path, load_model)
                # net_params = net_params[kwargs.get("model_name")]
                net_params = net_params[list(net_params.keys())[0]]
            except (KeyError, TypeError):
                try:
                    net_params = previous_load_ckpt(path, kwargs.get("model_name", None))
                except KeyError:
                    net_params = other_load_ckpt(path)
                    # This is required since other models were trained with slightly different varaible name
                    tmp = {}
                    for key, values in net_params.items():
                        key = key.replace("firstConvolution", "input_conv")
                        key = key.replace("contractingPath", "contracting_path")
                        key = key.replace("expandingPath", "expanding_path")
                        key = key.replace("lastConv", "output_conv")
                        tmp[key] = values
                    net_params = tmp 
        self.load_state_dict(net_params)

def get_backbone(name:str, **kwargs):
    """
    Creates the network instance

    :param model_path: A `str` of the model path

    :returns : A `nn.Module` of the network
    """
    if name == "unet": 
        backbone = UNet(in_channels=1, out_channels=2)
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, None
