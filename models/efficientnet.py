"""
EfficientNet model for Gravitational Wave Detection
Uses pre-trained EfficientNet with CQT spectrogram input
"""

import torch
import torch.nn as nn

try:
    import efficientnet_pytorch
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False


class GWEfficientNet(nn.Module):
    """
    EfficientNet for Gravitational Wave Detection.
    
    Uses pre-trained EfficientNet models (B0-B7) with modified classifier head
    for binary classification on CQT spectrograms.
    
    Input: (batch, 3, freq_bins, time_bins) - 3-channel CQT spectrogram
    Output: (batch,) - logits for binary classification
    """
    
    def __init__(self, config):
        super().__init__()
        
        if not EFFICIENTNET_AVAILABLE:
            raise ImportError(
                "efficientnet_pytorch is required for EfficientNet models. "
                "Install with: pip install efficientnet_pytorch"
            )
        
        model_name = getattr(config, 'model_name', 'efficientnet-b0')
        
        if config.pretrained:
            self.net = efficientnet_pytorch.EfficientNet.from_pretrained(model_name)
        else:
            self.net = efficientnet_pytorch.EfficientNet.from_name(model_name)
        
        n_features = self.net._fc.in_features
        self.net._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=n_features, out_features=1, bias=True)
        )
    
    def forward(self, x):
        out = self.net(x)
        if isinstance(out, tuple):
            out = out[0]
        return out.squeeze(-1) if out.dim() > 1 else out
