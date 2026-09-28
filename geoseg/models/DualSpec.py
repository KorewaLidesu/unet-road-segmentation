import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FrequencyBranchOptimized(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyBranchOptimized, self).__init__()
        
        self.amp_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.phase_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        
        self.amp_norm = nn.BatchNorm2d(in_channels)
        self.phase_norm = nn.BatchNorm2d(in_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        fft_out = torch.fft.rfft2(x)
        
        amp = torch.abs(fft_out)
        phase = torch.angle(fft_out)
        
        amp_proc = self.lrelu(self.amp_norm(self.amp_conv(amp)))
        phase_proc = self.lrelu(self.phase_norm(self.phase_conv(phase)))

        real_part = amp_proc * torch.cos(phase_proc)
        imag_part = amp_proc * torch.sin(phase_proc)
        
        reconstructed_fft = torch.complex(real_part, imag_part)
        
        x_freq = torch.fft.irfft2(reconstructed_fft, s=x.shape[2:]) 
        return x_freq

class SpatialBranchOptimized(nn.Module):
    def __init__(self, in_channels):
        super(SpatialBranchOptimized, self).__init__()

        self.conv_5x5_1 = DWConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_5x5_2 = DWConv(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv_5x5_3 = DWConv(in_channels, in_channels, kernel_size=7, padding=3)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_attn = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.conv_out = DWConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_5x5_1(x)
        x2 = self.conv_5x5_2(x)
        x3 = self.conv_5x5_3(x)
        
        x_sum = x1 + x2 + x3
        
        attention_weight = torch.sigmoid(self.conv_attn(self.pool(x_sum)))
        x = x_sum * attention_weight
        
        x = self.bn_out(self.conv_out(x))
        return x

class HybridFeatureFusionOptimized(nn.Module):
    def __init__(self, decoder_channels, fusion_channels, out_channels):
        super(HybridFeatureFusionOptimized, self).__init__()
        
        concat_channels = decoder_channels + fusion_channels
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.freq_branch = FrequencyBranchOptimized(out_channels)
        self.spatial_branch = SpatialBranchOptimized(out_channels)
        
        self.bn_out = nn.BatchNorm2d(out_channels)
        
    def forward(self, decoder_features, fusion_features):
        decoder_features = F.interpolate(decoder_features, size=fusion_features.shape[2:], mode='bilinear', align_corners=False)
        
        X_C = torch.cat([decoder_features, fusion_features], dim=1)
        X_reduced = self.bottleneck(X_C)
        
        X_Freq = self.freq_branch(X_reduced)
        X_Spatial = self.spatial_branch(X_reduced)
        
        X_combined = X_Freq + X_Spatial
        hybrid_features = self.bn_out(X_combined)
        
        return hybrid_features

class DualSpec(nn.Module):
    def __init__(self, num_classes=2, dim_list=[96, 192, 384, 768]):
        super(DualSpec, self).__init__()
        
        self.fusion3_2 = HybridFeatureFusionOptimized(
            decoder_channels=dim_list[3], 
            fusion_channels=dim_list[2], 
            out_channels=dim_list[2]
        )

        self.fusion2_1 = HybridFeatureFusionOptimized(
            decoder_channels=dim_list[2], 
            fusion_channels=dim_list[1], 
            out_channels=dim_list[1]
        )

        self.fusion1_0 = HybridFeatureFusionOptimized(
            decoder_channels=dim_list[1], 
            fusion_channels=dim_list[0], 
            out_channels=dim_list[0]
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(dim_list[0], dim_list[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_list[0], num_classes, kernel_size=1)
        )

    def forward(self, f0, f1, f2, f3):
        d2 = self.fusion3_2(decoder_features=f3, fusion_features=f2)
        d1 = self.fusion2_1(decoder_features=d2, fusion_features=f1)
        d0 = self.fusion1_0(decoder_features=d1, fusion_features=f0)
        logits = self.seg_head(d0)
        
        out = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        return out
