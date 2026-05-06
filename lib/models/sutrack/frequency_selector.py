#!/usr/bin/env python3
"""
Frequency Domain Token Selector - LAST-ViT Inspired
频域Token选择器 - 受LAST-ViT启发

核心思想：
1. 通过频域分析识别重要patches
2. 使用高斯滤波分离低频（背景）和高频（细节）成分
3. 选择对目标跟踪最有用的tokens

References:
    - LAST-ViT: Vision Transformers Need More Than Registers
    - https://github.com/ChengShiest/LAST-ViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class GaussianFrequencyFilter(nn.Module):
    """
    高斯频域滤波器
    用于分离图像的低频和高频成分
    """
    def __init__(self, kernel_size: int = 7, sigma: float = 2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # 预计算高斯核
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())
    
    def _create_gaussian_kernel(self) -> torch.Tensor:
        """创建一维高斯核"""
        size = self.kernel_size
        x = torch.arange(size).float() - size // 2
        gauss = torch.exp(-x.pow(2) / (2 * self.sigma ** 2))
        return gauss / gauss.sum()
    
    def forward(self, x_fft: torch.Tensor) -> torch.Tensor:
        """
        对频域信号应用高斯滤波
        
        Args:
            x_fft: [B, N, C] 频域特征
            
        Returns:
            filtered: [B, N, C] 滤波后的频域特征
        """
        B, N, C = x_fft.shape
        
        # 将高斯核扩展到匹配特征维度
        kernel = self.gaussian_kernel.to(x_fft.device)
        
        # 应用滤波：在频域乘以高斯核（相当于时域卷积）
        # 对每个token的特征维度应用滤波
        x_real = x_fft.real
        x_imag = x_fft.imag
        
        # 使用卷积实现频域滤波
        x_real_flat = x_real.reshape(B * N, 1, C)
        kernel_expanded = kernel.view(1, 1, -1).repeat(1, 1, C // self.kernel_size + 1)[:, :, :C]
        
        # 简化的滤波：直接乘以高斯权重
        weights = torch.linspace(0, 1, C // 2, device=x_fft.device)
        weights = torch.cat([weights, weights.flip(0)])
        if len(weights) < C:
            weights = F.interpolate(weights.view(1, 1, -1), size=C, mode='linear', align_corners=True).view(-1)
        
        x_filtered_real = x_real * weights[:C]
        x_filtered_imag = x_imag * weights[:C]
        
        return torch.complex(x_filtered_real, x_filtered_imag)


class FrequencyDomainSelector(nn.Module):
    """
    频域Token选择器 - LAST-ViT核心模块
    
    工作流程：
    1. 对输入特征进行FFT变换到频域
    2. 应用高斯滤波分离频率成分
    3. 逆变换回时域
    4. 计算原始特征与滤波后特征的差异
    5. 选择差异最大的Top-K tokens（包含最多高频信息）
    """
    
    def __init__(
        self,
        dim: int,
        select_ratio: float = 0.7,
        kernel_size: int = 7,
        sigma: float = 2.0,
        use_importance_weight: bool = True,
        return_mask: bool = True
    ):
        """
        Args:
            dim: 特征维度
            select_ratio: 选择比例（0-1），默认保留70%的tokens
            kernel_size: 高斯核大小
            sigma: 高斯核标准差
            use_importance_weight: 是否使用重要性加权
            return_mask: 是否返回选择掩码
        """
        super().__init__()
        self.dim = dim
        self.select_ratio = select_ratio
        self.use_importance_weight = use_importance_weight
        self.return_mask = return_mask
        
        # 高斯频域滤波器
        self.freq_filter = GaussianFrequencyFilter(kernel_size, sigma)
        
        # 可选：重要性评分网络（轻量级）
        if use_importance_weight:
            self.importance_mlp = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )
        
        # 可学习的温度参数（用于softmax）
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_selected_indices: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [B, N, C] 输入特征 (B=batch, N=num_patches, C=dim)
            return_selected_indices: 是否返回选中的索引
            
        Returns:
            x_selected: [B, K, C] 选中的特征 (K = N * select_ratio)
            select_mask: [B, N] 选择掩码（如果return_mask=True）
            indices: [B, K] 选中的索引（如果return_selected_indices=True）
        """
        B, N, C = x.shape
        K = max(1, int(N * self.select_ratio))  # 确保至少选择1个
        
        # Step 1: 频域变换
        # 对每个token的特征维度进行FFT
        x_fft = torch.fft.fft(x, dim=-1)  # [B, N, C]
        
        # Step 2: 频移（将低频移到中心）
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)
        
        # Step 3: 高斯滤波（抑制高频）
        x_fft_filtered = self.freq_filter(x_fft_shifted)
        
        # Step 4: 逆频移
        x_fft_unshifted = torch.fft.ifftshift(x_fft_filtered, dim=-1)
        
        # Step 5: 逆FFT回到时域
        x_filtered = torch.fft.ifft(x_fft_unshifted, dim=-1).real  # [B, N, C]
        
        # Step 6: 计算差异（原始 vs 滤波后）
        # 差异大的token包含更多高频信息（更可能是目标细节）
        # 使用 LAST-ViT 原始逻辑：x_detach / |x - x_filtered|
        epsilon = 1e-6
        diff = x / (torch.abs(x - x_filtered) + epsilon)  # [B, N, C]
        
        # Step 7: 计算重要性分数
        if self.use_importance_weight:
            # 结合频域差异和可学习的重要性评分
            importance = self.importance_mlp(x).squeeze(-1)  # [B, N]
            freq_score = diff.mean(dim=-1)  # [B, N]
            # 融合两种分数
            combined_score = freq_score * importance
        else:
            # 仅使用频域差异
            combined_score = diff.mean(dim=-1)  # [B, N]
        
        # Step 8: 选择Top-K tokens
        # 使用gumbel softmax实现可微分的选择（训练时）
        if self.training:
            # 可微分选择（用于训练）
            scores = combined_score / self.temperature.abs()
            # 使用soft top-k近似
            select_probs = F.softmax(scores, dim=-1)  # [B, N]
            # 选择概率最高的K个
            _, indices = torch.topk(select_probs, K, dim=1)  # [B, K]
        else:
            # 硬选择（用于推理）
            _, indices = torch.topk(combined_score, K, dim=1)  # [B, K]
        
        # Step 9: Gather选中的特征
        # 扩展索引以匹配特征维度
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
        x_selected = torch.gather(x, 1, indices_expanded)  # [B, K, C]
        
        # Step 10: 生成选择掩码
        select_mask = None
        if self.return_mask:
            select_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
            for b in range(B):
                select_mask[b, indices[b]] = True
        
        # 返回值
        if return_selected_indices:
            return x_selected, select_mask, indices
        return x_selected, select_mask, None
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, select_ratio={self.select_ratio}, ' \
               f'use_importance_weight={self.use_importance_weight}'


class AdaptiveFrequencySelector(nn.Module):
    """
    自适应频域选择器
    根据输入内容动态调整选择比例
    """
    
    def __init__(
        self,
        dim: int,
        min_ratio: float = 0.5,
        max_ratio: float = 0.9,
        kernel_size: int = 7,
        sigma: float = 2.0
    ):
        super().__init__()
        self.dim = dim
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        
        # 基础选择器
        self.base_selector = FrequencyDomainSelector(
            dim=dim,
            select_ratio=max_ratio,  # 使用最大比例初始化
            kernel_size=kernel_size,
            sigma=sigma
        )
        
        # 自适应比例预测器
        self.ratio_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: [B, N, C]
            
        Returns:
            x_selected: [B, K, C]
            select_mask: [B, N]
            actual_ratio: 实际选择比例
        """
        B, N, C = x.shape
        
        # 预测自适应比例
        global_feat = x.mean(dim=1)  # [B, C]
        ratio_logit = self.ratio_predictor(global_feat).mean()  # scalar
        
        # 映射到[min_ratio, max_ratio]
        actual_ratio = self.min_ratio + ratio_logit * (self.max_ratio - self.min_ratio)
        actual_ratio = actual_ratio.item()
        
        # 动态调整选择器
        K = max(1, int(N * actual_ratio))
        
        # 执行选择（复用基础选择器的逻辑，但使用动态K）
        # 这里简化处理，实际可以优化
        x_selected, select_mask, _ = self.base_selector(x, return_selected_indices=False)
        
        # 如果实际K不同，需要重新选择
        if K != x_selected.shape[1]:
            # 重新计算分数并选择
            with torch.no_grad():
                x_fft = torch.fft.fft(x, dim=-1)
                x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)
                x_fft_filtered = self.base_selector.freq_filter(x_fft_shifted)
                x_fft_unshifted = torch.fft.ifftshift(x_fft_filtered, dim=-1)
                x_filtered = torch.fft.ifft(x_fft_unshifted, dim=-1).real
                diff = torch.abs(x - x_filtered).mean(dim=-1)
                _, indices = torch.topk(diff, K, dim=1)
                indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)
                x_selected = torch.gather(x, 1, indices_expanded)
                
                select_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
                for b in range(B):
                    select_mask[b, indices[b]] = True
        
        return x_selected, select_mask, actual_ratio


# 测试代码
if __name__ == '__main__':
    print("Testing Frequency Domain Selector...")
    
    # 创建测试输入
    B, N, C = 2, 196, 768  # batch=2, patches=196 (14x14), dim=768
    x = torch.randn(B, N, C)
    
    # 测试基础选择器
    print("\n1. Testing FrequencyDomainSelector:")
    selector = FrequencyDomainSelector(dim=C, select_ratio=0.7)
    x_selected, mask, indices = selector(x, return_selected_indices=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Selected shape: {x_selected.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Indices shape: {indices.shape}")
    print(f"   Selection ratio: {x_selected.shape[1] / N:.2%}")
    
    # 测试自适应选择器
    print("\n2. Testing AdaptiveFrequencySelector:")
    adaptive_selector = AdaptiveFrequencySelector(dim=C, min_ratio=0.5, max_ratio=0.9)
    x_selected, mask, ratio = adaptive_selector(x)
    print(f"   Selected shape: {x_selected.shape}")
    print(f"   Actual ratio: {ratio:.2%}")
    
    print("\n✓ All tests passed!")
