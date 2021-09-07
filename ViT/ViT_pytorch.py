import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


# Patch embedding layer to make input image to transformer layer input
class PatchEmbedding(nn.Module):
    def __init__(self, channel_input=3, patch_size=16, embedding_dim=768, img_size=224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # By using einops rearrange, transform input image in (s1, s2) size patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            # Linear embedding of flattened patches as described in original paper
            nn.Linear(patch_size * patch_size * channel_input, embedding_dim)
        )
        # Add cls_tokens for better classification task which is learnable parameter
        # nn.Parameter() is not a layer but learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1,1, embedding_dim))
        # Define positional embedding for each patch
        # Positional embedding was set to learnable parameter
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, embedding_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        # Copy cls_tokens for corrresponding batch size
        # n : num of patches, e : embedding_dim
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


# Multi Head Attention layer to define transformer layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=8, dropout: float = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        # Compute query, key and value embedding by one matrix
        self.qkv = nn.Linear(embedding_dim, embedding_dim*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        # split keys, queries and values in num_heads
        # embedding dimension d is splited by d -> d * num_heads & n=num of patches
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        # Each tensor size is [batch, num_head, num_patches, splited embedding_dim] form
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # Tensor multiplication over embedding by using einsum : dot product between key & query perfomed
        # Result tensor is [batch, num_heads, num_query, num_key] form
        att = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            att.mask_fill(~mask, fill_value)

        scaling = self.embedding_dim ** (1/2)
        att = F.softmax(att, dim=-1) / scaling
        att = self.att_drop(att)
        # Since num_query = num_key = num_value = num_patches,
        # att = [batch, num_heads, num_query, num_key] form
        # values = [batch, num_head, num_patches, splited embedding_dim] form
        # Result tensor = [batch, num_heads, num_query, splited embedding_dim]
        # Compute weighted sum of key values with attention values
        out = torch.einsum('bhnk, bhkd -> bhnd ', att, values)
        # Concatenate each head's result to original embedding dim
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# Residual connection
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs) + res
        return x

# Position-wise FFN after self-attention
class FFN(nn.Sequential):
    def __init__(self, embedding_dim, expansion:int=4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(embedding_dim, expansion * embedding_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * embedding_dim, embedding_dim),
        )

# Define actual Transformer encoder block by using above components
class EncoderBlock(nn.Sequential):
    def __init__(self, embedding_dim=768, num_heads=8, drop_p: float = 0.,
                 expansion: int = 4, forward_drop_p: float = 0.1):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_dim),
                MultiHeadAttention(embedding_dim, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_dim),
                FFN(embedding_dim, expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
            )

# Head to perform classification
class Head(nn.Sequential):
    def __init__(self, embedding_dim=768, num_class=1000):
        super().__init__(Reduce('b n e -> b e', reduction='mean'),
                            nn.LayerNorm(embedding_dim), 
                            nn.Linear(embedding_dim, num_class))

# Define ViT model class
class ViT(nn.Module):
    def __init__(self, channel_input=3, patch_size=16, embedding_dim=768, 
                    img_size=224, depth=12, num_head=8, ffn_expansion=2,
                    drop_p=0.1, forward_drop_p=0.1, num_class=1000):
        super().__init__()
        self.depth = depth
        self.PatchEmbedding = PatchEmbedding(channel_input, patch_size, 
                                                embedding_dim, img_size)
        self.EncoderBlock = EncoderBlock(embedding_dim, num_head, drop_p, 
                                            ffn_expansion, forward_drop_p)
        self.Head = Head(embedding_dim, num_class)

    def forward(self, x):
        x = self.PatchEmbedding(x)
        for i in range(self.depth):
            x = self.EncoderBlock(x)
        x = self.Head(x)
        return x


