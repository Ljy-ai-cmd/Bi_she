import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.evaluation.environment import env_settings
import os
import json
import torch

class AntiUAVDataset(BaseDataset):
    """
    AntiUAV dataset for evaluation.
    
    args:
        modality: 'rgb', 'ir', or 'all'.
                  'rgb': Returns stacked RGB+ZeroIR images.
                  'ir': Returns stacked ZeroRGB+IR images.
                  'all': Returns stacked RGB+IR images.
    """
    def __init__(self, modality='all'):
        super().__init__()
        self.modality = modality
        self.base_path = self.env_settings.antiuav_path
        print(f"AntiUAVDataset init. Base path: {self.base_path}")
        self.sequence_list = self._get_sequence_list()
        print(f"Found {len(self.sequence_list)} sequences.")

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _get_sequence_list(self):
        # Load test.json or val.json
        json_path = os.path.join(self.base_path, 'label_new', 'test.json')
        if not os.path.exists(json_path):
             print(f"{json_path} not found, trying val.json")
             json_path = os.path.join(self.base_path, 'label_new', 'val.json')
             
        print(f"Loading json from: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        return list(data.keys())

    def _construct_sequence(self, sequence_name):
        # Path to sequence: base_path/test/sequence_name or base_path/val/sequence_name
        seq_path = os.path.join(self.base_path, 'test', sequence_name)
        if not os.path.exists(seq_path):
             seq_path = os.path.join(self.base_path, 'val', sequence_name)
             
        # Load annotations
        # Prefer visible.json, then infrared.json
        anno_path = os.path.join(seq_path, 'visible.json')
        if not os.path.exists(anno_path):
            anno_path = os.path.join(seq_path, 'infrared.json')
            
        ground_truth_rect = None
        target_visible = None
        
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
            
            if 'gt_rect' in anno_data:
                gt_rect = anno_data['gt_rect']
                cleaned_rect = []
                for r in gt_rect:
                    if isinstance(r, list) and len(r) == 4:
                        cleaned_rect.append(r)
                    else:
                        cleaned_rect.append([0, 0, 0, 0])
                ground_truth_rect = np.array(cleaned_rect, dtype=np.float32)
                
            if 'exist' in anno_data:
                target_visible = np.array(anno_data['exist'], dtype=np.int8)
        
        # Construct frames list
        # We need to find how many frames. 
        # Usually from annotation length.
        if ground_truth_rect is not None:
            num_frames = len(ground_truth_rect)
        else:
            # Count files? 
            # Assume visible folder exists
            vis_dir = os.path.join(seq_path, 'visible')
            if os.path.exists(vis_dir):
                num_frames = len([name for name in os.listdir(vis_dir) if name.endswith('.jpg')])
            else:
                num_frames = 0
        
        # Determine naming convention from first frame
        naming_style = 'default' # {frame_id:06d}.jpg
        
        test_path_1 = os.path.join(seq_path, 'visible', '000001.jpg')
        test_path_2 = os.path.join(seq_path, 'visible', 'visibleI0000.jpg')
        
        if os.path.exists(test_path_1):
            naming_style = 'style1'
        elif os.path.exists(test_path_2):
            naming_style = 'style2'
            
        frames = []
        for i in range(num_frames):
            frame_id = i + 1
            if naming_style == 'style1':
                rgb_path = os.path.join(seq_path, 'visible', f'{frame_id:06d}.jpg')
                ir_path = os.path.join(seq_path, 'infrared', f'{frame_id:06d}.jpg')
            else:
                # style2
                rgb_path = os.path.join(seq_path, 'visible', f'visibleI{i:04d}.jpg')
                ir_path = os.path.join(seq_path, 'infrared', f'infraredI{i:04d}.jpg')

            frames.append((rgb_path, ir_path, self.modality))

        dataset_name = 'antiuav'
        if self.modality in ['rgb', 'ir']:
            dataset_name = f'antiuav_{self.modality}'

        return Sequence(sequence_name, frames, dataset_name, ground_truth_rect, target_visible=target_visible)
