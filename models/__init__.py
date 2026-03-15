"""
Gravitational Wave Detection Models

Available models:
- EfficientNet: Pre-trained EfficientNet for CQT spectrograms
- ResNet: ResNet architecture for CQT spectrograms  
- Transformer: Vision Transformer (ViT) for CQT spectrograms
- SimpleCNN: Simple 2D CNN baseline
"""

from .efficientnet import GWEfficientNet
from .resnet import GWResNet
from .transformer import GWTransformer
from .simple_cnn import SimpleCNN

__all__ = [
    'GWEfficientNet',
    'GWResNet', 
    'GWTransformer',
    'SimpleCNN',
    'get_model'
]


def get_model(model_name, config):
    """
    Factory function to get model by name.
    
    Args:
        model_name: One of 'efficientnet-b0' to 'efficientnet-b7', 
                    'resnet18', 'resnet34', 'resnet50',
                    'transformer', 'transformer-small', 'transformer-large',
                    'simple_cnn'
        config: Config object with model parameters
    
    Returns:
        nn.Module: The requested model
    """
    model_name = model_name.lower()
    
    if model_name.startswith('efficientnet'):
        return GWEfficientNet(config)
    elif model_name.startswith('resnet'):
        return GWResNet(config)
    elif model_name.startswith('transformer'):
        return GWTransformer(config)
    elif model_name == 'simple_cnn':
        return SimpleCNN(config)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: efficientnet-b0~b7, resnet18/34/50, "
                        f"transformer/transformer-small/transformer-large, simple_cnn")
