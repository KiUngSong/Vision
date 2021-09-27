import torch
import torch.nn.functional as F
from torch import nn

from einops.layers.torch import Rearrange, Reduce

# Define Embedding layer of Patch
class PatchEmbedding(nn.Sequential):
    def __init__(self, channel_input=3, patch_size=16, embedding_dim=768, img_size=224):
        super().__init__(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * channel_input, embedding_dim)
        )

# MLP block with 2-FC layers
class MLP(nn.Sequential):
    def __init__(self, embed_dim, hidden_dim, drop_p=0.2):
        super().__init__(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop_p)
        )

# Define Mixer block consists of Token mixing MLP & Channel mixing MLP
class Mixerblock(nn.Module):
    def __init__(self, num_patch, embed_dim, token_dim, channel_dim, drop_p=0.2):
        super().__init__()
        # Define Token mixing MLP
        self.token_mix = nn.Sequential(nn.LayerNorm(embed_dim),
                # b=batch size, n=number of patch, d=embed_dim(dimension of patch embedding)
                Rearrange('b n d -> b d n'),
                MLP(num_patch, token_dim, drop_p),
                Rearrange('b d n -> b n d'))
        # Define Channel mixing MLP
        self.channel_mix = nn.Sequential(nn.LayerNorm(embed_dim),
                MLP(embed_dim, channel_dim, drop_p))
    
    def forward(self, x):
        x = x + self.token_mix(x)
        return x + self.channel_mix(x)

# Head to perform classification
class Head(nn.Sequential):
    def __init__(self, embed_dim=768, num_class=1000):
        super().__init__(Reduce('b c d -> b d', reduction='mean'),
                            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_class))


class MLPMixer(nn.Module):
    def __init__(self, channel_input=3, patch_size=16, embed_dim=768, img_size=224, 
                    depth=12, token_dim=256, channel_dim=2048, drop_p=0.2, num_class=1000):
        super().__init__()
        self.depth = depth
        self.PatchEmbedding = PatchEmbedding(channel_input, patch_size, embed_dim, img_size)
        self.Mixerblock = Mixerblock(int(img_size**2 / patch_size**2), embed_dim, token_dim, channel_dim, drop_p)
        self.Head = Head(embed_dim, num_class)

    def forward(self, x):
        x = self.PatchEmbedding(x)
        for i in range(self.depth):
            x = self.Mixerblock(x)
        x = self.Head(x)
        return x