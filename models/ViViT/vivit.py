# modified from https://github.com/rishikksh20/ViViT-pytorch

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'mean', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, prepend_cls_token=False,
                 projection_head=True, classifier_head=False,
                 proj_head_hidden_layer= 2048,
                 out_dim = 128):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.prepend_cls_token = prepend_cls_token
        self.classifier_head = classifier_head
        self.projection_head = projection_head

        if self.prepend_cls_token:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))

        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        if self.projection_head:
            print('==> setting up non-linear project heads')
            self.fc1 = nn.Linear(dim, proj_head_hidden_layer)
            self.bn_proj = nn.BatchNorm1d(proj_head_hidden_layer)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(proj_head_hidden_layer, out_dim)

        if self.classifier_head:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )


    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        if self.prepend_cls_token:
            cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
            x = torch.cat((cls_space_tokens, x), dim=2)
            x += self.pos_embedding[:, :, :(n + 1)]
        else:
            x += self.pos_embedding[:, :, :n]

        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')

        x = self.space_transformer(x)

        if self.prepend_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        x = rearrange(x, '(b t) ... -> b t ...', b=b)

        if self.prepend_cls_token:
            cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        if self.projection_head:
            h = self.fc1(x)
            h = self.bn_proj(h)
            h = self.relu(h)
            h = self.fc2(h)

        if self.classifier_head:
            h = self.mlp_head(x)

        if self.projection_head or self.classifier_head:
            return h
        else:
            return x




if __name__ == "__main__":
    
    img = torch.ones([1, 16, 3, 224, 224]).cuda()
    
    model = ViViT(224, 16, 100, 16).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    
    
