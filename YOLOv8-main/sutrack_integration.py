"""
SUTrack系统集成模块

功能说明：
    本模块提供YOLOv8检测器与SUTrack跟踪系统的无缝集成接口。
    主要功能包括：
    1. 与SUTrack数据格式的兼容转换
    2. 提供跟踪器工厂类，统一管理跟踪器创建
    3. 实现跟踪状态管理
    4. 提供批量处理接口

作者：毕业设计项目
日期：2026-04-20
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加YOLOv8路径
YOLOV8_ROOT = Path(__file__).parent
sys.path.insert(0, str(YOLOV8_ROOT))

# 导入YOLOv8初始化模块
try:
    from yolo_tracker_initializer import (
        SUTrackInitializer,
        DetectionResult,
        TrackInitialization,
        YOLOv8Detector,
        InfraredImagePreprocessor,
        create_tracker_initializer,
        initialize_tracking
    )
    logger.info("YOLOv8初始化模块导入成功")
except ImportError as e:
    logger.error(f"YOLOv8初始化模块导入失败: {e}")
    raise


class TrackState(Enum):
    """跟踪状态枚举"""
    INITIALIZING = "initializing"    # 初始化中
    TRACKING = "tracking"            # 跟踪中
    LOST = "lost"                    # 目标丢失
    REINITIALIZING = "reinitializing" # 重新初始化


@dataclass
class TrackStateInfo:
    """跟踪状态信息"""
    state: TrackState
    frame_id: int
    bbox: Optional[np.ndarray] = None
    confidence: float = 0.0
    lost_count: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SUTrackBox:
    """
    SUTrack格式的边界框
    
    属性:
        x1, y1: 左上角坐标
        x2, y2: 右下角坐标
        cx, cy: 中心点坐标
        w, h: 宽度和高度
        confidence: 置信度
    """
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float = field(init=False)
    cy: float = field(init=False)
    w: float = field(init=False)
    h: float = field(init=False)
    confidence: float = 1.0
    
    def __post_init__(self):
        """计算派生属性"""
        self.cx = (self.x1 + self.x2) / 2
        self.cy = (self.y1 + self.y2) / 2
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
    
    @classmethod
    def from_xyxy(cls, bbox: np.ndarray, confidence: float = 1.0) -> 'SUTrackBox':
        """从xyxy格式创建"""
        return cls(
            x1=float(bbox[0]),
            y1=float(bbox[1]),
            x2=float(bbox[2]),
            y2=float(bbox[3]),
            confidence=confidence
        )
    
    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float, confidence: float = 1.0) -> 'SUTrackBox':
        """从cxcywh格式创建"""
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return cls(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)
    
    def to_xyxy(self) -> np.ndarray:
        """转换为xyxy格式"""
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    def to_cxcywh(self) -> np.ndarray:
        """转换为cxcywh格式"""
        return np.array([self.cx, self.cy, self.w, self.h])
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'x1': self.x1, 'y1': self.y1,
            'x2': self.x2, 'y2': self.y2,
            'cx': self.cx, 'cy': self.cy,
            'w': self.w, 'h': self.h,
            'confidence': self.confidence
        }


class TrackAdapter:
    """
    跟踪数据适配器
    
    功能：将YOLOv8检测结果转换为SUTrack格式
    """
    
    @staticmethod
    def detection_to_sutrack_box(detection: DetectionResult) -> SUTrackBox:
        """
        将DetectionResult转换为SUTrackBox
        
        参数:
            detection: YOLOv8检测结果
            
        返回:
            sutrack_box: SUTrack格式的边界框
        """
        return SUTrackBox.from_xyxy(detection.bbox, detection.confidence)
    
    @staticmethod
    def initialization_to_sutrack_box(initialization: TrackInitialization) -> SUTrackBox:
        """
        将TrackInitialization转换为SUTrackBox
        
        参数:
            initialization: 跟踪初始化数据
            
        返回:
            sutrack_box: SUTrack格式的边界框
        """
        return SUTrackBox.from_xyxy(
            initialization.initial_bbox,
            initialization.confidence
        )
    
    @staticmethod
    def to_sutrack_format(initialization: TrackInitialization) -> Dict:
        """
        转换为SUTrack系统所需的完整格式
        
        参数:
            initialization: 跟踪初始化数据
            
        返回:
            sutrack_data: SUTrack格式的初始化数据
        """
        box = TrackAdapter.initialization_to_sutrack_box(initialization)
        
        return {
            'target_id': initialization.target_id,
            'init_bbox': box.to_xyxy().tolist(),
            'init_center': initialization.initial_center.tolist(),
            'init_size': initialization.initial_size.tolist(),
            'confidence': initialization.confidence,
            'frame_id': initialization.frame_id,
            'box_dict': box.to_dict(),
            'metadata': initialization.metadata
        }


class TrackerFactory:
    """
    跟踪器工厂类
    
    功能：统一管理跟踪器的创建和配置
    """
    
    _instance = None
    _initializers = {}
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def create_tracker(
        cls,
        tracker_id: str,
        model_path: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ) -> SUTrackInitializer:
        """
        创建或获取跟踪器
        
        参数:
            tracker_id: 跟踪器唯一标识
            model_path: 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            device: 计算设备
            
        返回:
            initializer: SUTrackInitializer实例
        """
        if tracker_id not in cls._initializers:
            logger.info(f"创建新的跟踪器: {tracker_id}")
            cls._initializers[tracker_id] = SUTrackInitializer(
                model_path=model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device
            )
        else:
            logger.info(f"使用已存在的跟踪器: {tracker_id}")
        
        return cls._initializers[tracker_id]
    
    @classmethod
    def get_tracker(cls, tracker_id: str) -> Optional[SUTrackInitializer]:
        """
        获取已创建的跟踪器
        
        参数:
            tracker_id: 跟踪器标识
            
        返回:
            initializer: SUTrackInitializer实例，不存在返回None
        """
        return cls._initializers.get(tracker_id)
    
    @classmethod
    def remove_tracker(cls, tracker_id: str) -> bool:
        """
        移除跟踪器
        
        参数:
            tracker_id: 跟踪器标识
            
        返回:
            success: 是否成功移除
        """
        if tracker_id in cls._initializers:
            del cls._initializers[tracker_id]
            logger.info(f"移除跟踪器: {tracker_id}")
            return True
        return False
    
    @classmethod
    def list_trackers(cls) -> List[str]:
        """
        列出所有跟踪器ID
        
        返回:
            tracker_ids: 跟踪器ID列表
        """
        return list(cls._initializers.keys())
    
    @classmethod
    def clear_all(cls):
        """清除所有跟踪器"""
        cls._initializers.clear()
        logger.info("清除所有跟踪器")


class TrackStateManager:
    """
    跟踪状态管理器
    
    功能：管理多个跟踪目标的状态
    """
    
    def __init__(self):
        """初始化状态管理器"""
        self.states = {}
        self.history = {}
        logger.info("跟踪状态管理器初始化完成")
    
    def register_target(self, target_id: int, init_bbox: np.ndarray, confidence: float):
        """
        注册新目标
        
        参数:
            target_id: 目标ID
            init_bbox: 初始边界框
            confidence: 初始置信度
        """
        self.states[target_id] = TrackStateInfo(
            state=TrackState.TRACKING,
            frame_id=0,
            bbox=init_bbox,
            confidence=confidence,
            lost_count=0
        )
        self.history[target_id] = []
        logger.info(f"注册新目标: ID={target_id}")
    
    def update_state(
        self, 
        target_id: int, 
        frame_id: int,
        bbox: np.ndarray = None,
        confidence: float = None,
        state: TrackState = None
    ):
        """
        更新目标状态
        
        参数:
            target_id: 目标ID
            frame_id: 当前帧ID
            bbox: 边界框 (可选)
            confidence: 置信度 (可选)
            state: 状态 (可选)
        """
        if target_id not in self.states:
            logger.warning(f"目标不存在: ID={target_id}")
            return
        
        state_info = self.states[target_id]
        
        # 保存历史状态
        self.history[target_id].append({
            'frame_id': state_info.frame_id,
            'bbox': state_info.bbox.copy() if state_info.bbox is not None else None,
            'confidence': state_info.confidence,
            'state': state_info.state
        })
        
        # 更新状态
        state_info.frame_id = frame_id
        if bbox is not None:
            state_info.bbox = bbox
        if confidence is not None:
            state_info.confidence = confidence
        if state is not None:
            state_info.state = state
        
        # 更新丢失计数
        if state == TrackState.LOST:
            state_info.lost_count += 1
        elif state == TrackState.TRACKING:
            state_info.lost_count = 0
    
    def get_state(self, target_id: int) -> Optional[TrackStateInfo]:
        """
        获取目标状态
        
        参数:
            target_id: 目标ID
            
        返回:
            state_info: 状态信息
        """
        return self.states.get(target_id)
    
    def get_history(self, target_id: int) -> List[Dict]:
        """
        获取目标历史轨迹
        
        参数:
            target_id: 目标ID
            
        返回:
            history: 历史状态列表
        """
        return self.history.get(target_id, [])
    
    def remove_target(self, target_id: int):
        """
        移除目标
        
        参数:
            target_id: 目标ID
        """
        if target_id in self.states:
            del self.states[target_id]
            del self.history[target_id]
            logger.info(f"移除目标: ID={target_id}")


class SUTrackIntegration:
    """
    SUTrack系统集成类
    
    功能：提供与SUTrack系统的完整集成接口
    """
    
    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        初始化集成模块
        
        参数:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            device: 计算设备
        """
        self.initializer = SUTrackInitializer(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        
        self.state_manager = TrackStateManager()
        self.adapter = TrackAdapter()
        
        logger.info("SUTrack集成模块初始化完成")
    
    def initialize_first_frame(
        self, 
        first_frame: np.ndarray,
        frame_id: int = 0
    ) -> Optional[Dict]:
        """
        初始化第一帧
        
        参数:
            first_frame: 第一帧图像
            frame_id: 帧ID
            
        返回:
            sutrack_data: SUTrack格式的初始化数据
        """
        # 执行YOLOv8检测和初始化
        init_result = self.initializer.initialize(first_frame, frame_id)
        
        if init_result is None:
            logger.error("第一帧初始化失败")
            return None
        
        # 注册到状态管理器
        self.state_manager.register_target(
            init_result.target_id,
            init_result.initial_bbox,
            init_result.confidence
        )
        
        # 转换为SUTrack格式
        sutrack_data = self.adapter.to_sutrack_format(init_result)
        
        return sutrack_data
    
    def initialize_from_path(
        self, 
        image_path: Union[str, Path],
        frame_id: int = 0
    ) -> Optional[Dict]:
        """
        从图像路径初始化
        
        参数:
            image_path: 图像路径
            frame_id: 帧ID
            
        返回:
            sutrack_data: SUTrack格式的初始化数据
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        return self.initialize_first_frame(image, frame_id)
    
    def get_initialization_info(self) -> Dict:
        """
        获取初始化信息
        
        返回:
            info: 初始化信息字典
        """
        return {
            'model_path': str(self.initializer.detector.model_path),
            'class_names': self.initializer.detector.class_names,
            'device': self.initializer.detector.device,
            'conf_threshold': self.initializer.detector.conf_threshold,
            'iou_threshold': self.initializer.detector.iou_threshold
        }


# ==================== 便捷函数 ====================

def quick_initialize(
    first_frame: np.ndarray,
    model_path: str = None,
    conf_threshold: float = 0.25,
    return_format: str = 'sutrack'
) -> Optional[Union[Dict, TrackInitialization]]:
    """
    快速初始化函数
    
    参数:
        first_frame: 第一帧图像
        model_path: 模型路径
        conf_threshold: 置信度阈值
        return_format: 返回格式 ('sutrack', 'raw')
        
    返回:
        result: 初始化结果
    """
    integration = SUTrackIntegration(
        model_path=model_path,
        conf_threshold=conf_threshold
    )
    
    if return_format == 'sutrack':
        return integration.initialize_first_frame(first_frame)
    else:
        return integration.initializer.initialize(first_frame)


def batch_initialize(
    image_paths: List[Union[str, Path]],
    model_path: str = None,
    conf_threshold: float = 0.25
) -> List[Optional[Dict]]:
    """
    批量初始化
    
    参数:
        image_paths: 图像路径列表
        model_path: 模型路径
        conf_threshold: 置信度阈值
        
    返回:
        results: 初始化结果列表
    """
    integration = SUTrackIntegration(
        model_path=model_path,
        conf_threshold=conf_threshold
    )
    
    results = []
    for i, image_path in enumerate(image_paths):
        try:
            result = integration.initialize_from_path(image_path, frame_id=i)
            results.append(result)
        except Exception as e:
            logger.error(f"初始化失败 {image_path}: {e}")
            results.append(None)
    
    return results


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """模块测试"""
    
    print("=" * 60)
    print("SUTrack系统集成模块测试")
    print("=" * 60)
    
    # 测试图像路径
    test_image_path = YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg"
    
    # 如果测试图像不存在，查找其他图像
    if not test_image_path.exists():
        test_dirs = [YOLOV8_ROOT / "test_img", YOLOV8_ROOT / "data"]
        test_image_path = None
        for test_dir in test_dirs:
            if test_dir.exists():
                image_files = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png"))
                if image_files:
                    test_image_path = image_files[0]
                    break
    
    if test_image_path is None or not test_image_path.exists():
        print("错误: 未找到测试图像")
        sys.exit(1)
    
    print(f"\n测试图像: {test_image_path}")
    
    # 1. 测试TrackAdapter
    print("\n[1] 测试TrackAdapter...")
    initializer = SUTrackInitializer()
    init_result = initializer.initialize_from_path(test_image_path)
    
    if init_result:
        adapter = TrackAdapter()
        sutrack_box = adapter.initialization_to_sutrack_box(init_result)
        print(f"    SUTrackBox创建成功:")
        print(f"    xyxy: {sutrack_box.to_xyxy()}")
        print(f"    cxcywh: {sutrack_box.to_cxcywh()}")
        
        sutrack_data = adapter.to_sutrack_format(init_result)
        print(f"    SUTrack数据格式: {list(sutrack_data.keys())}")
    
    # 2. 测试TrackerFactory
    print("\n[2] 测试TrackerFactory...")
    tracker1 = TrackerFactory.create_tracker("tracker_1")
    tracker2 = TrackerFactory.create_tracker("tracker_2")
    tracker1_again = TrackerFactory.create_tracker("tracker_1")  # 应该返回已存在的
    
    print(f"    创建的跟踪器: {TrackerFactory.list_trackers()}")
    print(f"    tracker_1 是同一个实例: {tracker1 is tracker1_again}")
    
    # 3. 测试TrackStateManager
    print("\n[3] 测试TrackStateManager...")
    state_manager = TrackStateManager()
    
    if init_result:
        state_manager.register_target(
            init_result.target_id,
            init_result.initial_bbox,
            init_result.confidence
        )
        
        # 更新状态
        state_manager.update_state(
            init_result.target_id,
            frame_id=1,
            bbox=init_result.initial_bbox + 10,  # 模拟移动
            confidence=0.95
        )
        
        state_info = state_manager.get_state(init_result.target_id)
        print(f"    目标状态: {state_info.state.value}")
        print(f"    历史记录数: {len(state_manager.get_history(init_result.target_id))}")
    
    # 4. 测试SUTrackIntegration
    print("\n[4] 测试SUTrackIntegration...")
    integration = SUTrackIntegration(conf_threshold=0.25)
    
    sutrack_result = integration.initialize_first_frame(
        cv2.imread(str(test_image_path)),
        frame_id=0
    )
    
    if sutrack_result:
        print(f"    SUTrack集成成功!")
        print(f"    目标ID: {sutrack_result['target_id']}")
        print(f"    初始边界框: {sutrack_result['init_bbox']}")
        print(f"    Box字典: {sutrack_result['box_dict']}")
        
        # 获取初始化信息
        info = integration.get_initialization_info()
        print(f"    模型信息: {info['model_path']}")
    
    # 5. 测试便捷函数
    print("\n[5] 测试便捷函数...")
    quick_result = quick_initialize(
        cv2.imread(str(test_image_path)),
        return_format='sutrack'
    )
    print(f"    快速初始化成功: {quick_result is not None}")
    
    # 清理
    TrackerFactory.clear_all()
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
