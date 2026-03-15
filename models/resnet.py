"""
ResNet model for Gravitational Wave Detection
Uses PyTorch's pre-trained ResNet with CQT spectrogram input
"""

import torch
import torch.nn as nn
import torchvision.models as models


class GWResNet(nn.Module):
    """
    ResNet for Gravitational Wave Detection.
    
    Uses pre-trained ResNet models (18, 34, 50, 101, 152) with modified classifier
    for binary classification on CQT spectrograms.
    
    Input: (batch, 3, freq_bins, time_bins) - 3-channel CQT spectrogram
    Output: (batch,) - logits for binary classification
    """
    
    RESNET_CONFIGS = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
        'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2),
    }
    
    def __init__(self, config):
        super().__init__()
        
        model_name = getattr(config, 'model_name', 'resnet50')
        pretrained = getattr(config, 'pretrained', True)
        
        if model_name not in self.RESNET_CONFIGS:
            raise ValueError(f"Unknown ResNet model: {model_name}. "
                           f"Available: {list(self.RESNET_CONFIGS.keys())}")
        
        model_fn, weights = self.RESNET_CONFIGS[model_name]
        
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)
        
        n_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(-1) if out.dim() > 1 else out
