import os
import numpy as np
import torch
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader_w_failsafe

class AntiUAVOptimizedDataset(BaseVideoDataset):
    """针对小显存优化的Anti-UAV数据集"""
    
    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe, split='train', 
                 max_frames_per_seq=50,  # 限制每序列最大帧数
                 cache_size=10):         # 缓存大小
        if root is None:
            root = os.path.join(os.path.dirname(__file__), '../../../data/Anti-UAV410')

        super().__init__('AntiUAV', root, image_loader)
        self.split = split
        self.max_frames_per_seq = max_frames_per_seq
        self.cache = {}
        self.cache_size = cache_size
        
        self._load_sequences()

    def get_frames(self, seq_id, frame_ids, anno=None):
        """优化版的帧获取，带缓存机制"""
        seq_info = self.sequence_list[seq_id]
        
        # 限制帧数
        if len(frame_ids) > self.max_frames_per_seq:
            frame_ids = frame_ids[:self.max_frames_per_seq]

        frames = []
        frame_anno = []
        
        for frame_id in frame_ids:
            cache_key = f"{seq_info['name']}_{frame_id}"
            
            if cache_key in self.cache:
                # 使用缓存
                frame, anno_data = self.cache[cache_key]
            else:
                # 加载新帧
                img_name = f'{frame_id + 1:06d}.jpg'
                img_path = os.path.join(seq_info['image_dir'], img_name)
                
                if os.path.exists(img_path):
                    frame = self.image_loader(img_path)
                    # 转换为灰度图（红外数据）
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
                    
                    if anno is None:
                        anno = self.get_sequence_info(seq_id)
                    
                    bbox = anno['bbox'][frame_id] if frame_id < len(anno['bbox']) else [0,0,0,0]
                    anno_data = {'bbox': bbox, 'valid': anno['valid'][frame_id]}
                    
                    # 缓存
                    if len(self.cache) < self.cache_size:
                        self.cache[cache_key] = (frame, anno_data)
                else:
                    frame = None
                    anno_data = {'bbox': [0,0,0,0], 'valid': False}
            
            frames.append(frame)
            frame_anno.append(anno_data)

        meta_info = {'object_class_name': 'uav', 'sequence': seq_info['name']}
        return frames, frame_anno, meta_info