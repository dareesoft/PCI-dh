import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn import Conv2d, BatchNorm2d, ReLU
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple
from einops import rearrange, repeat
from torchvision.models import VisionTransformer

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EnhancedOrientationAnalysis(nn.Module):
    def __init__(self, in_channels=256, num_angles=16):
        super().__init__()
        self.num_angles = num_angles
        
        # Learnable orientation filters
        self.adaptive_filters = nn.Parameter(
            torch.randn(num_angles, 1, 5, 5)
        )
        nn.init.kaiming_normal_(self.adaptive_filters)
        
        # Multi-scale processing
        self.scales = [1, 2, 4]
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 64, 3, padding=1, dilation=scale)
            for scale in self.scales
        ])
        
        # Self-attention for orientation features
        self.attention = MultiHeadAttention(64 * len(self.scales))
        
        # Final orientation classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(64 * len(self.scales), 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_angles, 1)
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        scale_features = []
        for conv in self.scale_convs:
            feat = conv(x)
            scale_features.append(feat)
        
        # Concatenate multi-scale features
        features = torch.cat(scale_features, dim=1)
        
        # Apply self-attention
        B, C, H, W = features.shape
        features = rearrange(features, 'b c h w -> b (h w) c')
        features = self.attention(features)
        features = rearrange(features, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Orientation classification
        orientation_logits = self.classifier(features)
        
        return {
            'orientation_map': orientation_logits,
            'orientation_probs': F.softmax(orientation_logits, dim=1)
        }

class CrossModalityAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.texture_to_gloss = MultiHeadAttention(dim)
        self.gloss_to_depth = MultiHeadAttention(dim)
        self.depth_to_texture = MultiHeadAttention(dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        
    def forward(self, texture_feat, gloss_feat, depth_feat):
        # Cross-modality attention
        texture_enhanced = self.texture_to_gloss(texture_feat)
        gloss_enhanced = self.gloss_to_depth(gloss_feat)
        depth_enhanced = self.depth_to_texture(depth_feat)
        
        # Feature fusion
        fused = torch.cat([texture_enhanced, gloss_enhanced, depth_enhanced], dim=-1)
        fused = self.fusion(fused)
        
        return fused

class EnhancedTextureAnalysis(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.scales = [3, 5, 7, 9]  # Multiple scales
        
        # Multi-scale convolutions
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, k, padding=k//2),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ) for k in self.scales
        ])
        
        # Self-attention module
        self.attention = MultiHeadAttention(64 * len(self.scales))
        
        # GLCM-inspired texture features
        self.glcm_features = nn.Sequential(
            nn.Conv2d(64 * len(self.scales), 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1)
        )
        
        # Texture classifier
        self.classifier = nn.Conv2d(64, 3, 1)  # Smooth, Normal, Rough
        
    def forward(self, x):
        # Multi-scale feature extraction
        scale_features = []
        for conv in self.scale_convs:
            feat = conv(x)
            scale_features.append(feat)
            
        # Concatenate features
        features = torch.cat(scale_features, dim=1)
        
        # Apply attention
        B, C, H, W = features.shape
        features = rearrange(features, 'b c h w -> b (h w) c')
        features = self.attention(features)
        features = rearrange(features, 'b (h w) c -> b c h w', h=H, w=W)
        
        # GLCM feature extraction
        texture_features = self.glcm_features(features)
        
        # Classification
        texture_map = self.classifier(texture_features)
        
        return texture_map

class TransformerTemporalAnalysis(nn.Module):
    def __init__(self, dim=256, num_heads=8, num_layers=6):
        super().__init__()
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(d_model=dim, dropout=0.1)
        
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4
            ),
            num_layers=num_layers
        )
        
        # Temporal memory bank
        self.memory_size = 64
        self.memory = nn.Parameter(torch.randn(self.memory_size, dim))
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        
    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        
        # Reshape for transformer
        x = rearrange(x_seq, 'b t c h w -> (b h w) t c')
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add memory tokens with positional encoding
        memory = repeat(self.memory, 'm d -> (b h w) m d', b=B, h=H, w=W)
        memory = self.pos_encoder(memory)
        x = torch.cat([memory, x], dim=1)
        
        # Apply transformer
        x = self.temporal_transformer(x)
        
        # Separate memory and features
        _, x = x[:, :self.memory_size], x[:, self.memory_size:]
        
        # Predict next frame
        pred = self.predictor(x[:, -1])
        
        # Reshape back
        x = rearrange(x, '(b h w) t c -> b t c h w', b=B, h=H, w=W)
        pred = rearrange(pred, '(b h w) c -> b c h w', b=B, h=H, w=W)
        
        return x, pred

class UncertaintyHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)
        self.uncertainty = nn.Conv2d(in_channels, num_classes, 1)
        
    def forward(self, x):
        x = self.dropout(x)
        pred = self.classifier(x)
        uncertainty = torch.exp(self.uncertainty(x))  # Aleatoric uncertainty
        return pred, uncertainty

class EnhancedDistressDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Add spatial positional encoding
        self.spatial_pos_encoder = PositionalEncoding(
            d_model=256,  # Same as embed_dim
            dropout=0.1,
            max_len=16*16  # patch_size=16 for 256x256 image
        )
        
        self.backbone = VisionTransformer(
            patch_size=16,
            embed_dim=256,
            num_heads=8,
            num_layers=12
        )
        
        # Enhanced analysis modules
        self.orientation = EnhancedOrientationAnalysis()
        self.texture = EnhancedTextureAnalysis()
        self.temporal = TransformerTemporalAnalysis()
        
        # Cross-modality attention
        self.cross_attention = CrossModalityAttention(dim=256)
        
        # Uncertainty estimation
        self.uncertainty = UncertaintyHead(256, num_classes=19)  # 19 distress types
        
    def forward(self, x, x_seq=None):
        # Extract features with positional encoding
        x = self.backbone(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.size(1))))
        x = self.spatial_pos_encoder(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        features = x
        
        # Parallel analysis
        orientation_results = self.orientation(features)
        texture_results = self.texture(features)
        
        # Temporal analysis if sequence is provided
        temporal_results = None
        if x_seq is not None:
            temporal_results = self.temporal(x_seq)
        
        # Cross-modality fusion
        fused_features = self.cross_attention(
            texture_results,
            orientation_results['orientation_map'],
            features
        )
        
        # Final predictions with uncertainty
        predictions, uncertainties = self.uncertainty(fused_features)
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'orientation': orientation_results,
            'texture': texture_results,
            'temporal': temporal_results
        }

# Loss function
class EnhancedMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.task_weights = nn.Parameter(torch.ones(4))
        
    def forward(self, predictions, targets, uncertainties):
        # Task-specific losses with uncertainty weighting
        classification_loss = F.cross_entropy(
            predictions['predictions'],
            targets['labels']
        )
        
        orientation_loss = F.mse_loss(
            predictions['orientation']['orientation_map'],
            targets['orientation']
        )
        
        texture_loss = F.mse_loss(
            predictions['texture'],
            targets['texture']
        )
        
        # Uncertainty-aware loss combination
        weights = F.softmax(self.task_weights, dim=0)
        total_loss = (
            weights[0] * classification_loss / uncertainties['classification'] +
            weights[1] * orientation_loss / uncertainties['orientation'] +
            weights[2] * texture_loss / uncertainties['texture'] +
            0.5 * (torch.log(uncertainties['classification']) + 
                   torch.log(uncertainties['orientation']) +
                   torch.log(uncertainties['texture']))
        )
        
        return total_loss