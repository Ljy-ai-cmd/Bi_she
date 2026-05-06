# --------------------------------------------------------
# Fast-iTPN with Selective Linear Attention (LaSt-ViT inspired)
# 选择性线性注意力改进版本
# --------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from functools import partial
import warnings
import math
import torch
import torch.nn as nn
from timm.models.registry import register_model
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, drop_path, trunc_normal_

from torch import Tensor, Size
from typing import Union, List

# 导入选择性注意力
from .selective_attention import SelectiveLinearAttention, SelectiveBlock


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


_shape_t = Union[int, List[int], Size]


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        if subln:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.norm = norm_layer(hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.subln = subln

    def forward(self, x):
        x = self.fc1(x)
        if self.subln:
            x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.SiLU, drop=0., subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        
        if subln:
            self.norm = norm_layer(hidden_features)
        self.subln = subln

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        if self.subln:
            hidden = self.norm(hidden)
        x = self.w3(hidden)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """标准注意力（保留用于对比）"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., window_size=None, 
                 use_decoupled_rel_pos_bias=False, attn_head_dim=None,
                 deepnorm=False, subln=False, use_linear_attn=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_linear_attn = use_linear_attn
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithSelective(nn.Module):
    """
    支持选择性注意力的Block
    可以选择使用标准注意力、线性注意力或选择性线性注意力
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None,
                 use_decoupled_rel_pos_bias=False, depth=None, postnorm=False,
                 deepnorm=False, subln=False, swiglu=False, naiveswiglu=False,
                 use_linear_attn=False, use_selective_attn=False, select_ratio=0.7):
        super().__init__()
        
        with_attn = num_heads > 0
        self.use_selective_attn = use_selective_attn
        
        self.norm1 = norm_layer(dim) if with_attn else None
        
        if with_attn:
            if use_selective_attn:
                # 使用选择性线性注意力
                self.attn = SelectiveLinearAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop,
                    select_ratio=select_ratio
                )
            elif use_linear_attn:
                # 使用普通线性注意力
                from .fastitpn import LinearAttention
                self.attn = LinearAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop
                )
            else:
                # 使用标准注意力
                self.attn = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop,
                    window_size=window_size,
                    use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                    attn_head_dim=attn_head_dim,
                    deepnorm=deepnorm, subln=subln
                )
        else:
            self.attn = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, hidden_features=mlp_hidden_dim,
                subln=subln, norm_layer=norm_layer
            )
        else:
            self.mlp = Mlp(
                in_features=dim, hidden_features=mlp_hidden_dim,
                subln=subln, norm_layer=norm_layer
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) if self.attn is not None else None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)
        self.postnorm = postnorm
        
        # 存储选择掩码用于可视化
        self.select_mask = None

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.use_selective_attn and hasattr(self.attn, 'importance_scorer'):
            # 选择性注意力前向
            if self.gamma_2 is None:
                if self.postnorm:
                    attn_out, self.select_mask = self.attn(x, rel_pos_bias)
                    x = x + self.drop_path(self.norm1(attn_out))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    attn_out, self.select_mask = self.attn(self.norm1(x), rel_pos_bias)
                    x = x + self.drop_path(attn_out)
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                if self.postnorm:
                    attn_out, self.select_mask = self.attn(x, rel_pos_bias)
                    x = x + self.drop_path(self.gamma_1 * self.norm1(attn_out))
                    x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
                else:
                    attn_out, self.select_mask = self.attn(self.norm1(x), rel_pos_bias)
                    x = x + self.drop_path(self.gamma_1 * attn_out)
                    x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            # 普通注意力前向
            if self.gamma_2 is None:
                if self.postnorm:
                    if self.attn is not None:
                        x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.deepnorm:
                    if self.attn is not None:
                        residual = x
                        x = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                        x = self.drop_path(x)
                        x = residual * self.alpha + x
                        x = self.norm1(x)
                    residual = x
                    x = self.mlp(x)
                    x = self.drop_path(x)
                    x = residual * self.alpha + x
                    x = self.norm2(x)
                else:
                    if self.attn is not None:
                        x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                if self.postnorm:
                    if self.attn is not None:
                        x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                    x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
                else:
                    if self.attn is not None:
                        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                    x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x


def test_selective_block():
    """测试选择性Block"""
    print("Testing BlockWithSelective...")
    
    B, N, C = 2, 196, 384
    x = torch.randn(B, N, C)
    
    # 测试选择性注意力
    block = BlockWithSelective(
        dim=C, num_heads=8, 
        use_selective_attn=True, 
        select_ratio=0.7
    )
    
    out = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Select mask: {block.select_mask.shape if block.select_mask is not None else None}")
    print("\n[OK] BlockWithSelective test passed!")


if __name__ == '__main__':
    test_selective_block()
