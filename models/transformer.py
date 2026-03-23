"""
Vision Transformer (ViT) model for Gravitational Wave Detection.

Supports:
- Custom lightweight ViT for transformer-small
- Torchvision pretrained ViT backbones for transformer/base/large
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class PatchEmbedding(nn.Module):
    """
    Convert 2D image into patch embeddings.
    
    Splits the image into non-overlapping patches and projects them
    to the embedding dimension.
    """
    
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # (B, embed_dim, H/P, W/P) -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-Forward Network (MLP) in Transformer."""
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""
    
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CustomTransformer(nn.Module):
    """
    Vision Transformer (ViT) for Gravitational Wave Detection.
    
    Processes CQT spectrograms using patch-based transformer architecture.
    Supports multiple configurations: small, base (default), large.
    
    Input: (batch, 3, freq_bins, time_bins) - 3-channel CQT spectrogram
    Output: (batch,) - logits for binary classification
    
    Model configurations:
    - transformer-small: 4 layers, 256 dim, 4 heads (~3M params)
    - transformer (base): 6 layers, 384 dim, 6 heads (~8M params)
    - transformer-large: 12 layers, 512 dim, 8 heads (~22M params)
    """
    
    TRANSFORMER_CONFIGS = {
        'transformer-small': {
            'embed_dim': 256,
            'n_layers': 4,
            'n_heads': 4,
            'mlp_ratio': 4.0,
            'patch_size': 8,
            'dropout': 0.1,
        },
        'transformer': {
            'embed_dim': 384,
            'n_layers': 6,
            'n_heads': 6,
            'mlp_ratio': 4.0,
            'patch_size': 8,
            'dropout': 0.1,
        },
        'transformer-base': {
            'embed_dim': 384,
            'n_layers': 6,
            'n_heads': 6,
            'mlp_ratio': 4.0,
            'patch_size': 8,
            'dropout': 0.1,
        },
        'transformer-large': {
            'embed_dim': 512,
            'n_layers': 12,
            'n_heads': 8,
            'mlp_ratio': 4.0,
            'patch_size': 8,
            'dropout': 0.2,
        },
    }
    
    def __init__(self, config):
        super().__init__()
        
        model_name = getattr(config, 'model_name', 'transformer')
        
        if model_name not in self.TRANSFORMER_CONFIGS:
            model_name = 'transformer'
        
        cfg = self.TRANSFORMER_CONFIGS[model_name]
        
        self.embed_dim = cfg['embed_dim']
        n_layers = cfg['n_layers']
        n_heads = cfg['n_heads']
        mlp_ratio = cfg['mlp_ratio']
        patch_size = cfg['patch_size']
        dropout = cfg['dropout']
        
        img_size = (69, 129)
        in_channels = 3
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, self.embed_dim)
        n_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(self.embed_dim)
        
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        cls_output = x[:, 0]
        
        out = self.head(cls_output)
        return out.squeeze(-1)


class TorchvisionViT(nn.Module):
    """Torchvision ViT wrapper with optional pretrained weights."""

    VIT_CONFIGS = {
        'transformer': ('vit_b_16', tv_models.ViT_B_16_Weights.IMAGENET1K_V1),
        'transformer-base': ('vit_b_16', tv_models.ViT_B_16_Weights.IMAGENET1K_V1),
        'transformer-large': ('vit_l_16', tv_models.ViT_L_16_Weights.IMAGENET1K_V1),
    }

    def __init__(self, config):
        super().__init__()

        model_name = getattr(config, 'model_name', 'transformer')
        pretrained = getattr(config, 'pretrained', True)

        if model_name not in self.VIT_CONFIGS:
            model_name = 'transformer'

        vit_name, weights_enum = self.VIT_CONFIGS[model_name]
        weights = weights_enum if pretrained else None

        if vit_name == 'vit_b_16':
            self.backbone = tv_models.vit_b_16(weights=weights)
        elif vit_name == 'vit_l_16':
            self.backbone = tv_models.vit_l_16(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT model: {vit_name}")

        hidden_dim = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(hidden_dim, 1)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.use_input_norm = pretrained

    def forward(self, x):
        # Resize CQT images to ViT input size.
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Match the distribution expected by pretrained ImageNet ViT weights.
        if self.use_input_norm:
            x = (x - self.imagenet_mean) / self.imagenet_std

        out = self.backbone(x)
        return out.squeeze(-1) if out.dim() > 1 else out


class GWTransformer(nn.Module):
    """
    Transformer model selector.

    - `transformer-small`: custom lightweight ViT from scratch
    - `transformer` / `transformer-base`: torchvision ViT-B/16
    - `transformer-large`: torchvision ViT-L/16
    """

    def __init__(self, config):
        super().__init__()

        model_name = getattr(config, 'model_name', 'transformer')

        if model_name == 'transformer-small':
            if getattr(config, 'pretrained', True):
                print("Warning: pretrained weights are not available for transformer-small; using random initialization.")
            self.model = CustomTransformer(config)
        else:
            self.model = TorchvisionViT(config)

    def forward(self, x):
        return self.model(x)
