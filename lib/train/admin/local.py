import os
import platform

class EnvironmentSettings:
    def __init__(self):
        # 检测操作系统
        is_windows = platform.system() == 'Windows'
        
        if is_windows:
            # Windows 本地路径
            base_dir = 'E:/biyesheji/SUTrack-main11'
        else:
            # Ubuntu 服务器路径
            base_dir = '/data/gcj/wch/SUTrack-main11'
        
        self.workspace_dir = base_dir    # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(base_dir, 'tensorboard')    # Directory for tensorboard files.
        self.pretrained_networks = os.path.join(base_dir, 'pretrained_networks')
        self.lasot_dir = os.path.join(base_dir, 'data/lasot')
        self.vasttrack_dir = os.path.join(base_dir, 'data/vasttrack')
        self.got10k_dir = os.path.join(base_dir, 'data/got10k/train')
        self.lasot_lmdb_dir = os.path.join(base_dir, 'data/lasot_lmdb')
        self.got10k_lmdb_dir = os.path.join(base_dir, 'data/got10k_lmdb')
        self.trackingnet_dir = os.path.join(base_dir, 'data/trackingnet')
        self.trackingnet_lmdb_dir = os.path.join(base_dir, 'data/trackingnet_lmdb')
        self.coco_dir = os.path.join(base_dir, 'data/coco')
        self.coco_lmdb_dir = os.path.join(base_dir, 'data/coco_lmdb')
        self.imagenet1k_dir = os.path.join(base_dir, 'data/imagenet1k')
        self.imagenet22k_dir = os.path.join(base_dir, 'data/imagenet22k')
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = os.path.join(base_dir, 'data/vid')
        self.imagenet_lmdb_dir = os.path.join(base_dir, 'data/vid_lmdb')
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.depthtrack_dir = os.path.join(base_dir, 'data/depthtrack/train')
        self.lasher_dir = os.path.join(base_dir, 'data/lasher/trainingset')
        self.visevent_dir = os.path.join(base_dir, 'data/visevent/train')
        self.refcoco_dir = os.path.join(base_dir, 'data/refcoco')
        self.tnl2k_dir = os.path.join(base_dir, 'data/tnl2k/train')
        self.otb99_dir = os.path.join(base_dir, 'data/otb_lang')
        
        # 添加Anti-UAV数据集路径配置 (指向处理后的数据)
        self.antiuav_dir = os.path.join(base_dir, 'data/AntI-UAV')
        self.antiuav_anno_dir = os.path.join(base_dir, 'data/AntI-UAV')
