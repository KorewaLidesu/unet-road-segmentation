import torch
import torch.nn as nn
import torch.nn.functional as F


class cSE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(cSE, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.squeeze(x)


class scSE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(scSE, self).__init__()

        self.c_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.s_att = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c_gate = self.c_att(x)
        c_out = x * c_gate.expand_as(x) 

        s_gate = self.s_att(x)
        s_out = x * s_gate.expand_as(x)

        return c_out + s_out

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, 
                                   groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.depthwise(x))
        x = self.bn2(self.pointwise(x))
        return x


class MAFM(nn.Module):
    def __init__(self, shallow_channels, deep_channels, out_channels, reduction=4):
        super(MAFM, self).__init__()

        self.cSE_shared = nn.Sequential(
            nn.Conv2d(shallow_channels, shallow_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(shallow_channels // reduction, shallow_channels, kernel_size=1, bias=False)
        )

        self.scSE_block = scSE(deep_channels, reduction)
        self.dsconv_att = DSConv(deep_channels, deep_channels, kernel_size=3, padding=1)

        if shallow_channels != deep_channels:
            self.s_to_d = nn.Conv2d(shallow_channels, deep_channels, kernel_size=1, bias=False)
        else:
            self.s_to_d = nn.Identity()
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(shallow_channels + deep_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DSConv(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

        if out_channels != deep_channels:
             self.align_res = nn.Conv2d(deep_channels, out_channels, 1, bias=False)
        else:
             self.align_res = nn.Identity()

    def forward(self, shallow_features, deep_features):
        X_S = shallow_features
        X_D = deep_features 

        X_S_avg = F.adaptive_avg_pool2d(X_S, 1)
        X_S_max = F.adaptive_max_pool2d(X_S, 1)
        
        A_S_raw = self.cSE_shared(X_S_max) + self.cSE_shared(X_S_avg)
        A_S_proj = self.s_to_d(A_S_raw)
        A_S = torch.sigmoid(A_S_proj)

        X_D_scse = self.scSE_block(X_D)
        X_D_dsconv = self.dsconv_att(X_D_scse)
        A_D = torch.sigmoid(X_D_dsconv) 

        X_D_att = X_D * A_S.expand_as(X_D) 
        X_D_att = X_D_att * A_D 

        X_D_att_up = F.interpolate(X_D_att, size=shallow_features.shape[2:], mode='bilinear', align_corners=True)

        X_concat = torch.cat([X_S, X_D_att_up], dim=1)
        X_refined = self.fusion_conv(X_concat)

        X_F = X_refined + self.align_res(X_D_att_up)

        return X_F
