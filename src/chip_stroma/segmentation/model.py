# ==============================================================================
# Script:           model.py
# Purpose:          LightningModule wrapping SMP U-Net
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import segmentation_models_pytorch as smp
import torch.nn as nn

def build_model(encoder_name:     str = "resnet34",
                encoder_weights:  str = "imagenet",
                decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
                in_channels:      int = 3,
                out_classes:      int = 1) -> nn.Module:
    """
    Factory function for smp U-Net vessel segmentation model.

    ResNet34 encoder selected for optimal capacity/overfitting tradeoff
    at n=24 patients (Raghu et al., NeurIPS, 2019; Talo et al.,
    Comput Biol Med, 2019). ImageNet pretraining improves convergence
    on small medical imaging datasets (Chen et al., CVPR, 2021).

    Factory pattern follows established research pipeline conventions
    for config-driven model instantiation (Goyal et al., FAIR, 2022;
    Bouthillier et al., NeurIPS, 2021).

    Parameters
    ----------
    encoder_name : str
        smp encoder backbone. Default: 'resnet34'.
    encoder_weights : str
        Pretrained weights source. Default: 'imagenet'.
    decoder_channels : tuple[int, ...]
        Feature channels per decoder block, high to low resolution.
        Length must equal encoder depth. Default: (256, 128, 64, 32, 16).
    in_channels : int
        Input image channels. Default: 3 (RGB).
    out_classes : int
        Output segmentation classes. Default: 1 (binary vessel mask).

    Returns
    -------
    nn.Module
        Initialized smp U-Net with pretrained encoder.
    """

    # Return the pretrained ImageNet encoder
    return smp.Unet(
        encoder_name     = encoder_name,
        encoder_weights  = encoder_weights,
        decoder_channels = decoder_channels,
        in_channels      = in_channels,
        classes          = out_classes,
        activation       = None,
    )