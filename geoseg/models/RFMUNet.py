import torch
import torch.nn as nn
import torch.nn.functional as F
from .ADAMamba import vanilla_vmamba_tiny
from .MAFM import MAFM
from .DualSpec import DualSpec


class ADAMambaEncoder(nn.Module):
    def __init__(self, pretrained=False, **encoder_kwargs):
        super(ADAMambaEncoder, self).__init__()
        self.encoder = vanilla_vmamba_tiny(pretrained=pretrained, **encoder_kwargs)

    def forward(self, x):
        x = self.encoder(x)
        for i in range(len(x)):
            x[i] = x[i].permute(0, 3, 1, 2).contiguous()
        x = [x[0], x[1], x[2], x[-1]]
        return x


class RFMUNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, **encoder_kwargs):
        """RFM-UNet.

        ``pretrained`` loads the VMamba-tiny ImageNet weights into the encoder
        (non-strict: the ADAMamba blocks add parameters VMamba does not have).
        ``encoder_kwargs`` are forwarded to ``VSSM`` -- ``use_checkpoint=True`` in
        particular trades ~35% step time for a large drop in activation memory,
        which is what makes 512x512 training fit on a 16GB Kaggle T4.
        """
        super(RFMUNet, self).__init__()
        self.encoder = ADAMambaEncoder(pretrained=pretrained, **encoder_kwargs)
        self.mafm1 = MAFM(shallow_channels=96, deep_channels=192, out_channels=96)
        self.mafm2 = MAFM(shallow_channels=192, deep_channels=384, out_channels=192)
        self.mafm3 = MAFM(shallow_channels=384, deep_channels=768, out_channels=384)
        self.decoder = DualSpec(num_classes=num_classes, dim_list=[96, 192, 384, 768])

    def forward(self, x):
        features = self.encoder(x)
        f1 = self.mafm1(features[0], features[1])
        f2 = self.mafm2(features[1], features[2])
        f3 = self.mafm3(features[2], features[3])
        output = self.decoder(f1, f2, f3, features[3])

        return output
