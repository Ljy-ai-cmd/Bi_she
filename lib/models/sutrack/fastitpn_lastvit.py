#!/usr/bin/env python3
"""
Fast-iTPN with LAST-ViT Frequency Domain Selection
融合 LAST-ViT 频域选择机制的 Fast-iTPN 编码器

核心创新：
1. 在 Fast-iTPN 中集成频域 Token 选择机制
2. 保留预训练权重兼容性（保守策略）
3. 针对 Anti-UAV 红外场景优化

References:
    - LAST-ViT: Vision Transformers Need More Than Registers
    - Fast-iTPN: Fast Vision Transformer with Hierarchical Attention
"""

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
from typing import Union, List, Tuple, Optional

# 导入频域选择器
from .frequency_selector import FrequencyDomainSelector, AdaptiveFrequencySelector

# 从 fastitpn 导入必要的类和函数
try:
    from .fastitpn import (
        Fast_iTPN, _cfg, DropPath, Mlp, SwiGLU,
        Attention, Block, PatchEmbed, RelativePositionBias
    )
except ImportError:
    # 如果导入失败，重新定义必要的类
    from .fastitpn_selective import (
        Fast_iTPN, _cfg, DropPath, Mlp, SwiGLU,
        Attention, Block, PatchEmbed, RelativePositionBias
    )


class Fast_iTPN_LASTViT(Fast_iTPN):
    """
    Fast-iTPN with LAST-ViT Frequency Domain Selection
    
    改进点：
    1. 在特征提取后添加频域选择模块
    2. 选择性地保留包含目标信息的 tokens
    3. 抑制背景干扰，提高跟踪精度
    
    预训练权重兼容性：
    - 基础 Fast-iTPN 部分可以加载预训练权重
    - 新增的频域选择模块需要从头训练
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=6,
        num_classes=1000,
        embed_dim=384,
        depth_stage1=1,
        depth_stage2=2,
        depth=12,
        num_heads=6,
        bridge_mlp_ratio=3.,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True,
        init_scale=0.001,
        lin_drop=0.1,
        lin_depth=2,
        lin_embed_dim=384 * 2,
        convmlp=False,
        naiveswiglu=False,
        subln=False,
        pos_type='interpolate',
        token_type_indicate=False,
        task_num=1,
        # LAST-ViT 特定参数
        use_lastvit_selection=True,
        select_ratio=0.7,
        freq_kernel_size=7,
        freq_sigma=2.0,
        use_adaptive_selection=False,
        apply_selection_after_stage=2,  # 在第几个stage后应用选择
        **kwargs
    ):
        """
        Args:
            use_lastvit_selection: 是否启用 LAST-ViT 频域选择
            select_ratio: Token 选择比例（0-1）
            freq_kernel_size: 高斯滤波核大小
            freq_sigma: 高斯核标准差
            use_adaptive_selection: 是否使用自适应选择比例
            apply_selection_after_stage: 在哪个 stage 后应用选择（1, 2, 或 3）
        """
        # 调用父类初始化
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth_stage1=depth_stage1,
            depth_stage2=depth_stage2,
            depth=depth,
            num_heads=num_heads,
            bridge_mlp_ratio=bridge_mlp_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            use_mean_pooling=use_mean_pooling,
            init_scale=init_scale,
            lin_drop=lin_drop,
            lin_depth=lin_depth,
            lin_embed_dim=lin_embed_dim,
            convmlp=convmlp,
            naiveswiglu=naiveswiglu,
            subln=subln,
            pos_type=pos_type,
            token_type_indicate=token_type_indicate,
            task_num=task_num,
            **kwargs
        )
        
        # LAST-ViT 配置
        self.use_lastvit_selection = use_lastvit_selection
        self.select_ratio = select_ratio
        self.apply_selection_after_stage = apply_selection_after_stage
        
        # 创建频域选择器
        if use_lastvit_selection:
            if use_adaptive_selection:
                self.freq_selector = AdaptiveFrequencySelector(
                    dim=embed_dim,
                    min_ratio=0.5,
                    max_ratio=select_ratio,
                    kernel_size=freq_kernel_size,
                    sigma=freq_sigma
                )
            else:
                self.freq_selector = FrequencyDomainSelector(
                    dim=embed_dim,
                    select_ratio=select_ratio,
                    kernel_size=freq_kernel_size,
                    sigma=freq_sigma,
                    use_importance_weight=True,
                    return_mask=True
                )
            
            # 注意：不使用额外的位置编码参数
            # 位置编码在 prepare_tokens_with_masks 中已经添加
            # 频域选择后不再额外添加位置编码
        
        # 存储选择掩码（用于可视化）
        self.last_selection_mask = None
        self.last_selection_ratio = None
    
    def apply_frequency_selection(
        self, 
        xz: torch.Tensor,
        num_template_tokens: int,
        num_search_tokens: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        对拼接后的模板和搜索特征应用频域选择
        
        Args:
            xz: [B, N, C] 拼接后的特征（模板 + 搜索）
            num_template_tokens: 模板token数量
            num_search_tokens: 搜索token数量
            
        Returns:
            xz_selected: [B, K, C] 选择后的特征
            select_mask: [B, N] 选择掩码
        """
        B, N, C = xz.shape
        
        # 分离模板和搜索特征
        x = xz[:, :num_search_tokens, :]  # 搜索区域
        z = xz[:, num_search_tokens:, :]  # 模板
        
        # 对搜索区域应用频域选择（模板通常较小，不选择）
        if num_search_tokens > 0:
            x_selected, x_mask, _ = self.freq_selector(
                x, return_selected_indices=True
            )
            
            # 计算实际选择比例
            self.last_selection_ratio = x_selected.shape[1] / num_search_tokens
            
            # 合并模板和选择后的搜索特征
            xz_selected = torch.cat([x_selected, z], dim=1)
            
            # 创建完整的选择掩码
            select_mask = torch.zeros(B, N, device=xz.device, dtype=torch.bool)
            select_mask[:, :num_search_tokens] = x_mask
            select_mask[:, num_search_tokens:] = True  # 模板全部保留
            
            self.last_selection_mask = select_mask
            
            return xz_selected, select_mask
        else:
            # 如果没有搜索token，返回原特征
            return xz, None
    
    def forward_features_with_selection(
        self,
        template_list,
        search_list,
        template_anno_list,
        text_src,
        task_index
    ):
        """
        带频域选择的前向传播
        """
        # 1. 准备tokens（使用父类方法）
        xz = self.prepare_tokens_with_masks(
            template_list, search_list, template_anno_list, text_src, task_index
        )
        xz = self.pos_drop(xz)
        
        # 2. 计算token数量
        num_search = len(search_list)
        num_template = len(template_list)
        num_search_tokens = self.num_patches_search * num_search
        num_template_tokens = self.num_patches_template * num_template
        
        # 3. 应用频域选择（如果启用）
        if self.use_lastvit_selection and self.training:
            xz, select_mask = self.apply_frequency_selection(
                xz, num_template_tokens, num_search_tokens
            )
            # 注意：位置编码已经在 prepare_tokens_with_masks 中添加
            # 频域选择后不额外添加位置编码
        
        # 4. 通过主要blocks
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks[-self.num_main_blocks:]:
            xz = checkpoint.checkpoint(blk, xz, rel_pos_bias) if self.grad_ckpt else blk(xz, rel_pos_bias)
        
        # 5. 归一化
        xz = self.norm(xz)
        
        if self.fc_norm is not None:
            return self.fc_norm(xz)
        return xz
    
    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        """
        前向传播
        """
        if self.use_lastvit_selection:
            xz = self.forward_features_with_selection(
                template_list, search_list, template_anno_list, text_src, task_index
            )
        else:
            # 使用标准前向传播
            xz = self.forward_features(
                template_list, search_list, template_anno_list, text_src, task_index
            )
        
        out = [xz]
        return out
    
    def get_selection_info(self) -> dict:
        """
        获取最后一次选择的信息（用于可视化）
        
        Returns:
            dict: 包含选择掩码和选择比例
        """
        return {
            'selection_mask': self.last_selection_mask,
            'selection_ratio': self.last_selection_ratio,
            'num_selected': self.last_selection_mask.sum().item() if self.last_selection_mask is not None else 0
        }


def load_pretrained_lastvit(
    model: Fast_iTPN_LASTViT,
    checkpoint_path: str,
    pos_type: str = 'interpolate',
    patchembed_init: str = 'halfcopy',
    freeze_base: bool = True
):
    """
    加载预训练权重到 LAST-ViT 模型
    支持部分加载（只加载形状匹配的参数）
    
    Args:
        model: Fast_iTPN_LASTViT 模型
        checkpoint_path: 预训练权重路径
        pos_type: 位置编码处理方式
        patchembed_init: patch embedding 初始化方式
        freeze_base: 是否冻结基础 Fast-iTPN 参数
    """
    
    # 1. 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 从 checkpoint 中提取状态字典
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 2. 获取模型当前状态字典
    model_state = model.state_dict()
    
    # 3. 筛选形状匹配的参数
    matched_state = {}
    unmatched_keys = []
    
    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                matched_state[key] = value
            else:
                unmatched_keys.append(f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}")
        else:
            unmatched_keys.append(f"{key}: not in model")
    
    # 4. 加载匹配的参数
    model.load_state_dict(matched_state, strict=False)
    
    # 5. 打印加载信息
    print(f"[LAST-ViT] Loaded pretrained weights from {checkpoint_path}")
    print(f"[LAST-ViT]  - Matched parameters: {len(matched_state)}/{len(state_dict)}")
    
    if unmatched_keys:
        print(f"[LAST-ViT]  - Unmatched parameters: {len(unmatched_keys)}")
        for key in unmatched_keys[:5]:  # 只显示前5个
            print(f"    - {key}")
        if len(unmatched_keys) > 5:
            print(f"    ... and {len(unmatched_keys) - 5} more")
    
    # 6. 冻结基础参数（可选）
    if freeze_base:
        # 冻结 patch_embed
        for param in model.patch_embed.parameters():
            param.requires_grad = False
        
        # 冻结 blocks
        for param in model.blocks.parameters():
            param.requires_grad = False
        
        # 冻结 norm
        for param in model.norm.parameters():
            param.requires_grad = False
        
        # 不冻结 freq_selector
        if model.use_lastvit_selection and hasattr(model, 'freq_selector'):
            for param in model.freq_selector.parameters():
                param.requires_grad = True
        
        print("[LAST-ViT]  - Frozen base Fast-iTPN parameters")
        print("[LAST-ViT]  - Frequency selector parameters are trainable")
    
    return model


# 注册模型
@register_model
def fastitpnb_lastvit(
    pretrained=False,
    pos_type='interpolate',
    pretrain_type='',
    patchembed_init='copy',
    use_lastvit_selection=True,
    select_ratio=0.7,
    **kwargs
):
    """
    Fast-iTPN Base with LAST-ViT selection
    """
    model = Fast_iTPN_LASTViT(
        patch_size=16,
        embed_dim=512,
        depth_stage1=2,
        depth_stage2=2,
        depth=24,
        num_heads=8,
        bridge_mlp_ratio=3.,
        mlp_ratio=3.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type=pos_type,
        use_lastvit_selection=use_lastvit_selection,
        select_ratio=select_ratio,
        **kwargs
    )
    model.default_cfg = _cfg()
    
    if pretrained and pretrain_type:
        load_pretrained_lastvit(
            model, pretrain_type, pos_type, patchembed_init, freeze_base=True
        )
    
    return model


@register_model
def fastitpns_lastvit(
    pretrained=False,
    pos_type='interpolate',
    pretrain_type='',
    patchembed_init='copy',
    use_lastvit_selection=True,
    select_ratio=0.7,
    **kwargs
):
    """
    Fast-iTPN Small with LAST-ViT selection
    """
    model = Fast_iTPN_LASTViT(
        patch_size=16,
        embed_dim=384,
        depth_stage1=2,
        depth_stage2=2,
        depth=20,
        num_heads=6,
        bridge_mlp_ratio=3.,
        mlp_ratio=3.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type=pos_type,
        use_lastvit_selection=use_lastvit_selection,
        select_ratio=select_ratio,
        **kwargs
    )
    model.default_cfg = _cfg()
    
    if pretrained and pretrain_type:
        load_pretrained_lastvit(
            model, pretrain_type, pos_type, patchembed_init, freeze_base=True
        )
    
    return model


@register_model
def fastitpnt_lastvit(
    pretrained=False,
    pos_type='interpolate',
    pretrain_type='',
    patchembed_init='copy',
    use_lastvit_selection=True,
    select_ratio=0.7,
    **kwargs
):
    """
    Fast-iTPN Tiny with LAST-ViT selection
    """
    model = Fast_iTPN_LASTViT(
        patch_size=16,
        embed_dim=384,
        depth_stage1=1,
        depth_stage2=1,
        depth=12,
        num_heads=6,
        bridge_mlp_ratio=3.,
        mlp_ratio=3.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type=pos_type,
        use_lastvit_selection=use_lastvit_selection,
        select_ratio=select_ratio,
        **kwargs
    )
    model.default_cfg = _cfg()
    
    if pretrained and pretrain_type:
        load_pretrained_lastvit(
            model, pretrain_type, pos_type, patchembed_init, freeze_base=True
        )
    
    return model


# 测试代码
if __name__ == '__main__':
    print("Testing Fast_iTPN_LASTViT...")
    
    # 创建测试输入
    B = 2
    template_list = [torch.randn(B, 6, 112, 112)]
    search_list = [torch.randn(B, 6, 224, 224)]
    template_anno_list = [torch.tensor([[56, 56, 50, 50]]).float().repeat(B, 1)]
    
    print("\n1. Testing Fast_iTPN_LASTViT (base):")
    model = fastitpnb_lastvit(
        use_lastvit_selection=True,
        select_ratio=0.7
    )
    output = model(template_list, search_list, template_anno_list, None, 0)
    print(f"   Output shape: {output[0].shape}")
    
    # 获取选择信息
    info = model.get_selection_info()
    print(f"   Selection ratio: {info['selection_ratio']:.2%}")
    print(f"   Num selected: {info['num_selected']}")
    
    print("\n2. Testing without selection:")
    model_no_sel = fastitpnb_lastvit(
        use_lastvit_selection=False
    )
    output_no_sel = model_no_sel(template_list, search_list, template_anno_list, None, 0)
    print(f"   Output shape: {output_no_sel[0].shape}")
    
    print("\n✓ All tests passed!")
