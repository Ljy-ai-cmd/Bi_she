#!/usr/bin/env python3
"""
Anti-UAV Dataset with CLAHE Enhancement
支持CLAHE预处理的红外图像增强版本
"""

import os
import numpy as np
import torch
import json
import cv2
from lib.train.dataset.antiuav import AntiUAVDataset
from lib.train.data.clahe_processor import CLAHEProcessor


class AntiUAVDatasetCLAHE(AntiUAVDataset):
    """
    Anti-UAV数据集CLAHE增强版本
    
    特点:
    - 只对红外通道应用CLAHE增强
    - RGB通道保持不变
    - 可配置CLAHE参数
    """
    
    def __init__(self, root=None, image_loader=None, split='train',
                 multi_modal_vision=False, multi_modal_language=False, use_nlp=False,
                 use_clahe=True, clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
        """
        Args:
            use_clahe: 是否启用CLAHE增强
            clahe_clip_limit: CLAHE对比度限制
            clahe_grid_size: CLAHE网格大小
        """
        super().__init__(root, image_loader, split, 
                        multi_modal_vision, multi_modal_language, use_nlp)
        
        self.use_clahe = use_clahe
        if use_clahe:
            self.clahe_processor = CLAHEProcessor(
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_grid_size,
                apply_to_ir=True,  # 只对红外增强
                apply_to_rgb=False  # RGB不变
            )
            print(f"[CLAHE] Enabled with clip_limit={clahe_clip_limit}, grid_size={clahe_grid_size}")
        else:
            self.clahe_processor = None
            print("[CLAHE] Disabled")
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        """
        获取帧，应用CLAHE增强
        """
        # 调用父类方法获取原始帧
        frame_list, anno_frames, object_meta = super().get_frames(seq_id, frame_ids, anno)
        
        # 应用CLAHE增强（只对红外通道）
        if self.use_clahe and self.clahe_processor is not None:
            enhanced_frames = []
            for frame in frame_list:
                if frame is not None:
                    enhanced = self.clahe_processor(frame)
                    enhanced_frames.append(enhanced)
                else:
                    enhanced_frames.append(None)
            frame_list = enhanced_frames
        
        return frame_list, anno_frames, object_meta


def test_clahe_dataset():
    """测试CLAHE数据集"""
    print("Testing AntiUAVDatasetCLAHE...")
    
    # 创建测试实例
    try:
        dataset = AntiUAVDatasetCLAHE(
            use_clahe=True,
            clahe_clip_limit=2.0,
            clahe_grid_size=(8, 8)
        )
        print(f"[OK] Dataset created with {len(dataset)} sequences")
        
        # 测试CLAHE处理器
        test_img = np.random.rand(224, 224, 6).astype(np.float32)
        enhanced = dataset.clahe_processor(test_img)
        
        print(f"[OK] CLAHE processing test passed")
        print(f"    Input shape: {test_img.shape}")
        print(f"    Output shape: {enhanced.shape}")
        
    except Exception as e:
        print(f"[Error] {e}")


if __name__ == '__main__':
    test_clahe_dataset()
