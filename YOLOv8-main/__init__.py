"""
YOLOv8红外目标检测与SUTrack跟踪初始化模块

本模块提供基于YOLOv8的红外无人机目标检测和SUTrack跟踪系统初始化功能。

主要组件:
    - yolo_tracker_initializer: 核心初始化模块
    - sutrack_integration: SUTrack系统集成模块

快速开始:
    >>> from YOLOv8_main import SUTrackInitializer
    >>> initializer = SUTrackInitializer()
    >>> result = initializer.initialize(first_frame)
    
    >>> from YOLOv8_main import quick_initialize
    >>> result = quick_initialize(first_frame)

作者: 毕业设计项目
日期: 2026-04-20
"""

__version__ = "1.0.0"
__author__ = "Graduation Project"

# 导入核心类
from .yolo_tracker_initializer import (
    SUTrackInitializer,
    YOLOv8Detector,
    InfraredImagePreprocessor,
    DetectionResult,
    TrackInitialization,
    create_tracker_initializer,
    initialize_tracking
)

# 导入集成模块
from .sutrack_integration import (
    SUTrackIntegration,
    SUTrackBox,
    TrackAdapter,
    TrackerFactory,
    TrackStateManager,
    TrackState,
    quick_initialize,
    batch_initialize
)

__all__ = [
    # 核心类
    'SUTrackInitializer',
    'YOLOv8Detector',
    'InfraredImagePreprocessor',
    'DetectionResult',
    'TrackInitialization',
    
    # 集成类
    'SUTrackIntegration',
    'SUTrackBox',
    'TrackAdapter',
    'TrackerFactory',
    'TrackStateManager',
    'TrackState',
    
    # 便捷函数
    'create_tracker_initializer',
    'initialize_tracking',
    'quick_initialize',
    'batch_initialize',
]
