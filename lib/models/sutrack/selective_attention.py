#!/usr/bin/env python3
"""
Selective Linear Attention Module
受LaSt-ViT启发的选择性注意力机制简化实现

核心思想：
1. 计算每个patch的重要性分数
2. 选择Top-K重要的patches
3. 只对选中的patches计算线性注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchImportanceScorer(nn.Module):
    """
    Patch重要性评分器
    基于内容自适应地评估每个patch的重要性
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 用于计算重要性分数的投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # 重要性评分网络（轻量级）
        self.importance_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()  # 输出0-1的重要性分数
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, C] - 输入特征
        Returns:
            importance: [B, N] - 每个patch的重要性分数
        """
        B, N, C = x.shape
        
        # 计算每个patch的自包含重要性
        # 方法：通过轻量级MLP直接预测
        importance = self.importance_mlp(x).squeeze(-1)  # [B, N]
        
        return importance


class SelectiveLinearAttention(nn.Module):
    """
    选择性线性注意力
    结合LaSt-ViT的选择性思想和线性注意力的效率优势
    
    特点：
    1. 先筛选重要patches（LaSt-ViT思想）
    2. 对选中patches使用线性注意力（高效计算）
    3. 保持O(N)复杂度，同时提升特征质量
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, 
                 attn_drop=0., proj_drop=0.,
                 select_ratio=0.7,  # 选择比例，默认保留70%的patches
                 use_importance_weight=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.select_ratio = select_ratio
        self.use_importance_weight = use_importance_weight
        
        # Patch重要性评分器
        self.importance_scorer = PatchImportanceScorer(dim, num_heads, qkv_bias)
        
        # 线性注意力的QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        """
        Args:
            x: [B, N, C] - 输入特征
            rel_pos_bias: 相对位置偏置（可选）
            attn_mask: 注意力掩码（可选）
        Returns:
            out: [B, N, C] - 输出特征
            select_mask: [B, N] - 选择掩码（用于可视化）
        """
        B, N, C = x.shape
        
        # Step 1: 计算每个patch的重要性分数 [B, N]
        importance = self.importance_scorer(x)
        
        # Step 2: 选择Top-K重要的patches
        k = int(N * self.select_ratio)  # 选择的patch数量
        
        # 获取Top-K的索引
        _, top_k_indices = torch.topk(importance, k, dim=-1, largest=True, sorted=False)
        
        # 创建选择掩码 [B, N]
        select_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        select_mask.scatter_(1, top_k_indices, True)
        
        # Step 3: 对选中的patches计算线性注意力
        # QKV投影
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
        
        # 应用选择掩码 - 只保留选中的patches的QKV
        # 未选中的patches的QKV设为0（不参与计算）
        q_masked = q * select_mask.unsqueeze(1).unsqueeze(-1)  # [B, heads, N, head_dim]
        k_masked = k * select_mask.unsqueeze(1).unsqueeze(-1)  # [B, heads, N, head_dim]
        v_masked = v * select_mask.unsqueeze(1).unsqueeze(-1)
        
        # Step 4: 线性注意力计算（只对选中patches）
        # 使用ELU+1核函数
        q_masked = F.elu(q_masked) + 1
        k_masked = F.elu(k_masked) + 1
        
        # 线性注意力核心: K^T @ V (O(N)复杂度)
        kv = torch.einsum('bhsk,bhsv->bhkv', k_masked, v_masked)  # [B, heads, head_dim, head_dim]
        
        # 计算归一化因子
        k_sum = k_masked.sum(dim=2, keepdim=True)  # [B, heads, 1, head_dim]
        z = 1 / (torch.einsum('bhqd,bhkd->bhq', q_masked, k_sum) + 1e-6)  # [B, heads, N]
        
        # 计算输出
        out = torch.einsum('bhqd,bhkd->bhqd', q_masked, kv)  # [B, heads, N, head_dim]
        out = out * z.unsqueeze(-1)  # [B, heads, N, head_dim]
        
        # 合并heads
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # 最终投影
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, select_mask


class SelectiveBlock(nn.Module):
    """
    选择性Transformer Block
    替换标准的Block，使用选择性线性注意力
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 select_ratio=0.7):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelectiveLinearAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            select_ratio=select_ratio
        )
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # 存储选择掩码用于可视化
        self.select_mask = None
    
    def forward(self, x, rel_pos_bias=None):
        # 选择性注意力
        attn_out, self.select_mask = self.attn(self.norm1(x), rel_pos_bias)
        x = x + self.drop_path(attn_out)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


def test_selective_attention():
    """测试选择性注意力"""
    print("Testing Selective Linear Attention...")
    
    B, N, C = 2, 196, 384  # batch=2, patches=196(14x14), dim=384
    x = torch.randn(B, N, C)
    
    # 创建模块
    attn = SelectiveLinearAttention(dim=C, num_heads=8, select_ratio=0.7)
    
    # 前向传播
    out, mask = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Select mask shape: {mask.shape}")
    print(f"Selected patches: {mask.sum(dim=1).float().mean():.0f} / {N}")
    print(f"Select ratio: {mask.sum(dim=1).float().mean() / N * 100:.1f}%")
    
    # 验证输出
    assert out.shape == x.shape, "Output shape mismatch!"
    assert mask.shape == (B, N), "Mask shape mismatch!"
    
    print("\n[OK] Selective Linear Attention test passed!")


if __name__ == '__main__':
    test_selective_attention()
