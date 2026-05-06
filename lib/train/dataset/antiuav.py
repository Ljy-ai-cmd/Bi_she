import os
import numpy as np
import torch
import json
import cv2
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader_w_failsafe
from lib.train.dataset.depth_utils import get_x_frame
from lib.train.admin import env_settings


class AntiUAVDataset(BaseVideoDataset):
    """ Anti-UAV dataset for training. """

    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe, split='train', 
                 multi_modal_vision=False, multi_modal_language=False, use_nlp=False,
                 infrared_only=False):
        """
        args:
            root - Path to the Anti-UAV dataset. If None, use default path
            image_loader - Image loader function
            split - Dataset split: 'train', 'val', or 'test'
            multi_modal_vision - Whether to use multi-modal vision data
            multi_modal_language - Whether to use multi-modal language data
            use_nlp - Whether to use NLP features
            infrared_only - When single-modal, load IR instead of visible (default: False)
        """
        if root is None:
            root = env_settings().antiuav_dir

        super().__init__('AntiUAV', root, image_loader)
        self.split = split
        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language
        self.use_nlp = use_nlp
        self.infrared_only = infrared_only

        # Load sequence list
        self._load_sequences()

    def get_name(self):
        return 'AntiUAV'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def _load_sequences(self):
        """Load sequences for the specified split."""
        split_dirs = [
            os.path.join(self.root, self.split),
            os.path.join(self.root, 'train'),
            os.path.join(self.root, 'val'),
            os.path.join(self.root, 'test'),
        ]
        split_dir = next((d for d in split_dirs if os.path.isdir(d)), None)
        if split_dir is None:
            print(f"Warning: Dataset directory {os.path.join(self.root, self.split)} not found.")
            return

        dir_seq_names = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

        label_path = os.path.join(self.root, 'label_new', f'{self.split}.json')
        label_seq_names = None
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            label_seq_names = list(label_data.keys())

        if label_seq_names is not None:
            dir_seq_set = set(dir_seq_names)
            seq_names = [n for n in label_seq_names if n in dir_seq_set]
            if len(seq_names) == 0:
                seq_names = dir_seq_names
        else:
            seq_names = dir_seq_names

        for seq_name in seq_names:
            seq_dir_candidates = [
                os.path.join(self.root, self.split, seq_name),
                os.path.join(self.root, 'train', seq_name),
                os.path.join(self.root, 'val', seq_name),
                os.path.join(self.root, 'test', seq_name),
            ]
            seq_dir = next((p for p in seq_dir_candidates if os.path.exists(p)), None)
            if seq_dir is None:
                continue

            anno_path = os.path.join(seq_dir, 'visible.json')
            if not os.path.exists(anno_path):
                anno_path = os.path.join(seq_dir, 'infrared.json')
            if not os.path.exists(anno_path):
                continue

            naming_style = self._detect_naming_style(seq_dir)
            self.sequence_list.append({
                'name': seq_name,
                'anno_path': anno_path,
                'image_dir': seq_dir,
                'naming_style': naming_style,
            })

    @staticmethod
    def _detect_naming_style(seq_dir: str):
        vis_dir = os.path.join(seq_dir, 'visible')
        if os.path.exists(os.path.join(vis_dir, '000001.jpg')):
            return 'style1_1based_6d'
        if os.path.exists(os.path.join(vis_dir, '000000.jpg')):
            return 'style0_0based_6d'
        if os.path.exists(os.path.join(vis_dir, 'visibleI0000.jpg')):
            return 'style2_0based_visibleI'
        return 'style2_0based_visibleI'

    @staticmethod
    def _resolve_rgb_ir_paths(seq_dir: str, idx: int, naming_style: str):
        vis_dir = os.path.join(seq_dir, 'visible')
        ir_dir = os.path.join(seq_dir, 'infrared')

        if naming_style == 'style1_1based_6d':
            fid = idx + 1
            rgb_path = os.path.join(vis_dir, f'{fid:06d}.jpg')
            ir_path = os.path.join(ir_dir, f'{fid:06d}.jpg')
        elif naming_style == 'style0_0based_6d':
            rgb_path = os.path.join(vis_dir, f'{idx:06d}.jpg')
            ir_path = os.path.join(ir_dir, f'{idx:06d}.jpg')
        else:
            rgb_path = os.path.join(vis_dir, f'visibleI{idx:04d}.jpg')
            ir_path = os.path.join(ir_dir, f'infraredI{idx:04d}.jpg')

        if not os.path.exists(rgb_path):
            alt = os.path.join(vis_dir, f'visibleI{idx:04d}.jpg')
            if os.path.exists(alt):
                rgb_path = alt
        if not os.path.exists(ir_path):
            alt = os.path.join(ir_dir, f'infraredI{idx:04d}.jpg')
            if os.path.exists(alt):
                ir_path = alt

        return rgb_path, ir_path

    @staticmethod
    def _load_ir_as_6ch(ir_path: str, fallback_path: str = None):
        ir = cv2.imread(ir_path) if os.path.exists(ir_path) else None
        if ir is None and fallback_path is not None:
            ir = cv2.imread(fallback_path)
        if ir is None:
            return None
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
        return np.concatenate((ir, ir), axis=2)

    @staticmethod
    def _safe_fuse_rgb_ir(rgb_path: str, ir_path: str):
        rgb = cv2.imread(rgb_path) if os.path.exists(rgb_path) else None
        ir = cv2.imread(ir_path) if os.path.exists(ir_path) else None

        if rgb is None and ir is None:
            return None

        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if ir is not None:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

        if rgb is None:
            rgb = np.zeros_like(ir)
        if ir is None:
            ir = np.zeros_like(rgb)

        if rgb.shape[:2] != ir.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))

        return np.concatenate((rgb, ir), axis=2)

    def get_sequence_info(self, seq_id):
        """Get sequence information."""
        # Ensure seq_id is int
        if isinstance(seq_id, str):
            try:
                seq_id = int(seq_id)
            except ValueError:
                seq_id = 0
        
        if seq_id < 0 or seq_id >= len(self.sequence_list):
            seq_id = 0
        
        seq_info = self.sequence_list[seq_id]
    
        # Load annotations
        anno_path = seq_info['anno_path']
        with open(anno_path, 'r') as f:
            data = json.load(f)
            
        # Parse annotations from JSON
        # Format: {"exist": [1, 1, ...], "gt_rect": [[x, y, w, h], ...]}
        
        if 'gt_rect' in data:
            gt_rect = data['gt_rect']
            cleaned_rect = []
            for r in gt_rect:
                if isinstance(r, list) and len(r) == 4:
                    cleaned_rect.append(r)
                else:
                    cleaned_rect.append([0, 0, 0, 0])
            bboxes = torch.tensor(cleaned_rect, dtype=torch.float32)
        else:
            # Fallback or error
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            
        if 'exist' in data:
            valid = torch.tensor(data['exist'], dtype=torch.bool)
        else:
            valid = torch.ones(len(bboxes), dtype=torch.bool)
            
        visible = valid.clone()
    
        return {
            'bbox': bboxes,
            'visible': visible,
            'valid': valid
        }

    def get_frames(self, seq_id, frame_ids, anno=None):
        """Get frames from sequence."""
        if isinstance(seq_id, str):
            try:
                seq_id = int(seq_id)
            except ValueError:
                seq_id = 0
        
        if seq_id < 0 or seq_id >= len(self.sequence_list):
            seq_id = 0
        
        seq_info = self.sequence_list[seq_id]
    
        if anno is None:
            anno = self.get_sequence_info(seq_id)
    
        # Load frames
        frame_list = []

        naming_style = seq_info.get('naming_style', 'style2_0based_visibleI')
        for frame_id in frame_ids:
            idx = int(frame_id)
            if self.multi_modal_vision:
                rgb_path, ir_path = self._resolve_rgb_ir_paths(seq_info['image_dir'], idx, naming_style)
                if self.infrared_only:
                    frame = self.image_loader(ir_path)
                    if frame is not None:
                        frame = np.concatenate((frame, frame), axis=-1)
                else:
                    frame = get_x_frame(rgb_path, ir_path, dtype='rgbrgb')
                frame_list.append(frame)
            else:
                # Single modality
                rgb_path, ir_path = self._resolve_rgb_ir_paths(seq_info['image_dir'], idx, naming_style)
                if self.infrared_only:
                    img_path = ir_path if os.path.exists(ir_path) else rgb_path
                else:
                    img_path = rgb_path if os.path.exists(rgb_path) else ir_path
                
                # Load image
                if os.path.exists(img_path):
                    frame = self.image_loader(img_path)
                    frame_list.append(frame)
                else:
                    frame_list.append(None)
    
        # Create annotation frames dictionary
        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                # Handle case where frame_id exceeds annotation length
                # This shouldn't happen if sampler is correct, but for safety
                valid_ids = [min(int(f_id), len(value)-1) for f_id in frame_ids]
                anno_frames[key] = [value[f_id, ...].clone() for f_id in valid_ids]
    
        # Create object meta information
        from collections import OrderedDict
        object_meta = OrderedDict({
            'object_class_name': 'uav',
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })
    
        return frame_list, anno_frames, object_meta
