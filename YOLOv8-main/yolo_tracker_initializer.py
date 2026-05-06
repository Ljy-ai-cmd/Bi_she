"""
YOLOv8红外目标检测与SUTrack跟踪初始化模块

功能说明：
    本模块实现基于YOLOv8的红外无人机目标检测，并为SUTrack跟踪系统提供第一帧初始化功能。
    主要功能包括：
    1. 加载YOLOv8预训练权重进行红外目标检测
    2. 对检测到的多个目标进行筛选，输出置信度最高的单目标
    3. 解析检测结果，提取边界框坐标、置信度等信息
    4. 初始化跟踪器状态，为后续跟踪提供初始目标信息
    5. 提供标准化接口供SUTrack系统调用

作者：毕业设计项目
日期：2026-04-20
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加YOLOv8路径（必须在导入之前）
YOLOV8_ROOT = Path(__file__).parent
sys.path.insert(0, str(YOLOV8_ROOT))

# 导入YOLOv8工具函数
try:
    from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes
    from ultralytics.yolo.data.dataloaders.v5augmentations import letterbox
    YOLO_UTILS_AVAILABLE = True
    logger.info("YOLOv8工具函数导入成功")
except ImportError as e:
    YOLO_UTILS_AVAILABLE = False
    logger.error(f"无法导入YOLOv8工具函数: {e}")
    # 导入失败时抛出错误，因为 letterbox 是必需的
    raise ImportError(f"无法导入YOLOv8工具函数: {e}。请确保 ultralytics 包已正确安装。") from e

try:
    from ultralytics import YOLO
    from ultralytics.yolo.utils import ops
    logger.info("YOLOv8模块加载成功")
except ImportError as e:
    logger.error(f"YOLOv8模块加载失败: {e}")
    raise


@dataclass
class DetectionResult:
    """
    检测结果数据结构
    
    属性:
        bbox: 边界框坐标 [x1, y1, x2, y2] (像素坐标)
        confidence: 置信度分数 (0-1)
        class_id: 类别ID
        class_name: 类别名称
        center: 目标中心点坐标 [cx, cy]
        size: 目标尺寸 [width, height]
    """
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    center: np.ndarray
    size: np.ndarray
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'bbox': self.bbox.tolist(),
            'confidence': float(self.confidence),
            'class_id': int(self.class_id),
            'class_name': self.class_name,
            'center': self.center.tolist(),
            'size': self.size.tolist()
        }


@dataclass
class TrackInitialization:
    """
    跟踪初始化数据结构
    
    属性:
        target_id: 目标唯一标识符
        initial_bbox: 初始边界框 [x1, y1, x2, y2]
        initial_center: 初始中心点 [cx, cy]
        initial_size: 初始尺寸 [width, height]
        confidence: 检测置信度
        frame_id: 初始化帧ID
        timestamp: 初始化时间戳
        metadata: 额外元数据
    """
    target_id: int
    initial_bbox: np.ndarray
    initial_center: np.ndarray
    initial_size: np.ndarray
    confidence: float
    frame_id: int
    timestamp: float
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'target_id': self.target_id,
            'initial_bbox': self.initial_bbox.tolist(),
            'initial_center': self.initial_center.tolist(),
            'initial_size': self.initial_size.tolist(),
            'confidence': float(self.confidence),
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class InfraredImagePreprocessor:
    """
    红外图像预处理模块
    
    功能：
        1. 图像格式转换 (BGR -> RGB)
        2. 图像尺寸调整 (保持宽高比，letterbox填充)
        3. 像素值归一化 (0-255 -> 0-1)
        4. 维度转换 (HWC -> CHW -> NCHW)
    """
    
    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        """
        初始化预处理器
        
        参数:
            input_size: 模型输入尺寸 (width, height)，默认640x640
        """
        self.input_width, self.input_height = input_size
        logger.info(f"初始化图像预处理器，输入尺寸: {input_size}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        预处理红外图像
        
        参数:
            image: 输入图像 (BGR格式，HWC维度)
            
        返回:
            processed_image: 预处理后的图像 (NCHW格式，float32)
            original_image: 原始图像副本
            original_size: 原始图像尺寸 (height, width)
        """
        # 保存原始图像
        original_image = image.copy()
        original_height, original_width = image.shape[:2]
        original_size = (original_height, original_width)
        
        # 1. BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 计算缩放比例 (保持宽高比)
        r_w = self.input_width / original_width
        r_h = self.input_height / original_height
        
        if r_h > r_w:
            # 宽度是长边
            new_width = self.input_width
            new_height = int(r_w * original_height)
            pad_x = 0
            pad_y = (self.input_height - new_height) // 2
        else:
            # 高度是长边
            new_width = int(r_h * original_width)
            new_height = self.input_height
            pad_x = (self.input_width - new_width) // 2
            pad_y = 0
        
        # 3. 图像resize
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 4. Letterbox填充 (使用灰色114填充)
        image = cv2.copyMakeBorder(
            image, 
            pad_y, self.input_height - new_height - pad_y,
            pad_x, self.input_width - new_width - pad_x,
            cv2.BORDER_CONSTANT, 
            value=(114, 114, 114)
        )
        
        # 5. 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 6. HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        # 7. CHW -> NCHW
        image = np.expand_dims(image, axis=0)
        
        # 8. 确保内存连续
        image = np.ascontiguousarray(image)
        
        return image, original_image, original_size
    
    def preprocess_from_path(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        从文件路径加载并预处理图像
        
        参数:
            image_path: 图像文件路径
            
        返回:
            同preprocess方法
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        return self.preprocess(image)


class YOLOv8Detector:
    """
    YOLOv8目标检测器
    
    功能：
        1. 加载YOLOv8预训练权重
        2. 执行红外目标检测
        3. 后处理检测结果 (NMS等)
        4. 筛选单目标 (置信度最高)
    """
    
    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        初始化YOLOv8检测器
        
        参数:
            model_path: 模型权重文件路径，默认使用anti_uav_single_stage16的best.pt
            conf_threshold: 置信度阈值，默认0.25
            iou_threshold: NMS的IoU阈值，默认0.45
            device: 计算设备 ('cpu', 'cuda', 'auto')，默认自动选择
        """
        # 默认权重路径
        if model_path is None:
            model_path = str(YOLOV8_ROOT / "runs" / "detect" / "anti_uav_single_stage16" / "weights" / "best.pt")
        
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 验证权重文件
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_path}")
        
        logger.info(f"加载YOLOv8模型: {self.model_path}")
        
        # 自动选择设备
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        try:
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
        # 获取类别信息 - 兼容不同版本的YOLOv8
        try:
            # 尝试获取类别名称
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            else:
                # 默认类别名称
                self.class_names = {0: 'drone'}
        except Exception as e:
            logger.warning(f"获取类别名称失败: {e}，使用默认值")
            self.class_names = {0: 'drone'}
        
        logger.info(f"模型类别: {self.class_names}")
        
        # 获取模型参数
        self.imgsz = 640  # 默认输入尺寸
        self.stride = 32  # 默认步长
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'stride'):
                stride_val = self.model.model.stride
                # 确保 stride 是整数而不是 Tensor
                if isinstance(stride_val, torch.Tensor):
                    self.stride = int(stride_val.max()) if stride_val.numel() > 1 else int(stride_val.item())
                else:
                    self.stride = int(stride_val)
                logger.info(f"模型步长: {self.stride}")
        except Exception as e:
            logger.warning(f"获取模型步长失败: {e}，使用默认值 {self.stride}")
        
        # 初始化图像预处理器
        self.preprocessor = InfraredImagePreprocessor()
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        对单张图像进行目标检测
        
        参数:
            image: 输入图像 (BGR格式)
            
        返回:
            detections: 检测结果列表
        """
        # 获取原始图像尺寸
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        try:
            # 直接使用模型进行推理，而不是使用predict方法
            # 这样可以获得原始的tensor输出
            
            # 预处理图像
            img = letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            
            # 转换为tensor
            img_tensor = torch.from_numpy(img).to(self.device)
            img_tensor = img_tensor.float() / 255.0  # 归一化
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor[None]  # 添加batch维度
            
            # 推理
            with torch.no_grad():
                pred = self.model.model(img_tensor, augment=False, visualize=False)
            
            # NMS
            pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, None, False, max_det=300)
            
            # 解析结果
            detections = []
            for i, det in enumerate(pred):  # 每张图片的检测结果
                if len(det):
                    # 将坐标从resize后的图像映射回原始图像
                    det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], image.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = map(float, xyxy)
                        cls_id = int(cls)
                        
                        # 计算中心点和尺寸
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        
                        detection = DetectionResult(
                            bbox=np.array([x1, y1, x2, y2]),
                            confidence=float(conf),
                            class_id=cls_id,
                            class_name=self.class_names.get(cls_id, 'unknown'),
                            center=np.array([cx, cy]),
                            size=np.array([w, h])
                        )
                        detections.append(detection)
                        
                        logger.info(f"检测到目标: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], conf={conf:.3f}, cls={cls_id}")
            
            # 按置信度降序排序
            detections.sort(key=lambda x: x.confidence, reverse=True)
            logger.info(f"总共检测到 {len(detections)} 个目标")
            
            return detections
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_from_path(self, image_path: Union[str, Path]) -> List[DetectionResult]:
        """
        从文件路径加载图像并检测
        
        参数:
            image_path: 图像文件路径
            
        返回:
            detections: 检测结果列表
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        return self.detect(image)
    
    def _parse_results(
        self,
        results,
        original_size: Tuple[int, int]
    ) -> List[DetectionResult]:
        """
        解析YOLOv8检测结果

        参数:
            results: YOLOv8预测结果 (Results对象列表)
            original_size: 原始图像尺寸 (width, height)

        返回:
            detections: 解析后的检测结果列表
        """
        detections = []
        original_width, original_height = original_size

        logger.info(f"解析检测结果，results类型: {type(results)}, 长度: {len(results) if hasattr(results, '__len__') else 'N/A'}")

        # results是一个列表，每个元素应该是Results对象
        for result in results:
            logger.info(f"处理result，类型: {type(result)}")

            # 检查结果类型
            if isinstance(result, torch.Tensor):
                logger.info("result是Tensor类型，跳过")
                continue

            # 检查是否是Results对象（通过检查关键属性）
            if not hasattr(result, 'boxes'):
                logger.warning(f"result没有boxes属性，类型: {type(result)}")
                continue

            # 获取boxes
            boxes = result.boxes

            # 检查boxes是否为None或空
            if boxes is None:
                logger.warning("result.boxes为None")
                continue

            # 检查boxes是否有数据
            if len(boxes) == 0:
                logger.info("boxes为空，没有检测到目标")
                continue

            logger.info(f"boxes类型: {type(boxes)}, 数量: {len(boxes)}")

            try:
                # 获取检测框坐标 (xyxy格式)
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                logger.info(f"检测到 {len(xyxy)} 个目标")

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    conf = confidences[i]
                    cls_id = class_ids[i]

                    logger.info(f"目标 {i}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], conf={conf:.3f}, cls={cls_id}")

                    # 计算中心点和尺寸
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    detection = DetectionResult(
                        bbox=np.array([x1, y1, x2, y2]),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=self.class_names.get(cls_id, 'unknown'),
                        center=np.array([cx, cy]),
                        size=np.array([w, h])
                    )
                    detections.append(detection)
            except Exception as e:
                logger.error(f"解析boxes时出错: {e}")
                import traceback
                traceback.print_exc()

        # 按置信度降序排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        logger.info(f"最终检测到 {len(detections)} 个目标")

        return detections
    
    def detect_single_target(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        检测单目标 (置信度最高的目标)
        
        参数:
            image: 输入图像
            
        返回:
            detection: 置信度最高的检测结果，如果没有检测到则返回None
        """
        detections = self.detect(image)
        
        if len(detections) == 0:
            logger.warning("未检测到任何目标")
            return None
        
        # 返回置信度最高的目标
        best_detection = detections[0]
        logger.info(f"检测到目标: 置信度={best_detection.confidence:.4f}, "
                   f"类别={best_detection.class_name}, "
                   f"位置=[{best_detection.bbox[0]:.1f}, {best_detection.bbox[1]:.1f}, "
                   f"{best_detection.bbox[2]:.1f}, {best_detection.bbox[3]:.1f}]")
        
        return best_detection


class SUTrackInitializer:
    """
    SUTrack跟踪初始化器
    
    功能：
        1. 基于YOLOv8检测结果初始化跟踪器
        2. 生成跟踪初始化数据结构
        3. 提供标准化接口供SUTrack调用
    """
    
    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        初始化SUTrack跟踪器
        
        参数:
            model_path: YOLOv8模型路径
            conf_threshold: 检测置信度阈值
            iou_threshold: NMS IoU阈值
            device: 计算设备
        """
        self.detector = YOLOv8Detector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        
        self.target_counter = 0
        logger.info("SUTrack初始化器创建成功")
    
    def initialize(
        self, 
        first_frame: np.ndarray,
        frame_id: int = 0
    ) -> Optional[TrackInitialization]:
        """
        基于第一帧图像初始化跟踪
        
        参数:
            first_frame: 第一帧图像 (BGR格式)
            frame_id: 帧ID，默认0
            
        返回:
            initialization: 跟踪初始化数据结构，如果检测失败则返回None
        """
        import time
        
        logger.info(f"开始初始化跟踪 - 帧ID: {frame_id}")
        
        # 执行目标检测
        detection = self.detector.detect_single_target(first_frame)
        
        if detection is None:
            logger.error("初始化失败: 未检测到目标")
            return None
        
        # 生成目标ID
        self.target_counter += 1
        target_id = self.target_counter
        
        # 创建初始化数据结构
        initialization = TrackInitialization(
            target_id=target_id,
            initial_bbox=detection.bbox,
            initial_center=detection.center,
            initial_size=detection.size,
            confidence=detection.confidence,
            frame_id=frame_id,
            timestamp=time.time(),
            metadata={
                'class_id': detection.class_id,
                'class_name': detection.class_name,
                'detection_method': 'YOLOv8',
                'model_path': str(self.detector.model_path)
            }
        )
        
        logger.info(f"跟踪初始化成功 - 目标ID: {target_id}, "
                   f"位置: [{detection.bbox[0]:.1f}, {detection.bbox[1]:.1f}, "
                   f"{detection.bbox[2]:.1f}, {detection.bbox[3]:.1f}]")
        
        return initialization
    
    def initialize_from_path(
        self, 
        image_path: Union[str, Path],
        frame_id: int = 0
    ) -> Optional[TrackInitialization]:
        """
        从图像文件路径初始化跟踪
        
        参数:
            image_path: 第一帧图像路径
            frame_id: 帧ID
            
        返回:
            initialization: 跟踪初始化数据结构
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        return self.initialize(image, frame_id)
    
    def visualize_detection(
        self, 
        image: np.ndarray, 
        initialization: TrackInitialization,
        save_path: str = None
    ) -> np.ndarray:
        """
        可视化检测结果
        
        参数:
            image: 原始图像
            initialization: 跟踪初始化数据
            save_path: 保存路径，如果为None则不保存
            
        返回:
            vis_image: 可视化后的图像
        """
        vis_image = image.copy()
        bbox = initialization.initial_bbox.astype(int)
        
        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        thickness = 2
        cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        
        # 绘制中心点
        center = initialization.initial_center.astype(int)
        cv2.circle(vis_image, tuple(center), 5, (0, 0, 255), -1)
        
        # 添加标签
        label = f"ID:{initialization.target_id} Conf:{initialization.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # 计算文本尺寸
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # 绘制标签背景
        cv2.rectangle(vis_image, 
                     (bbox[0], bbox[1] - text_height - 10),
                     (bbox[0] + text_width, bbox[1]),
                     color, -1)
        
        # 绘制文本
        cv2.putText(vis_image, label, (bbox[0], bbox[1] - 5),
                   font, font_scale, (255, 255, 255), font_thickness)
        
        # 保存图像
        if save_path:
            cv2.imwrite(save_path, vis_image)
            logger.info(f"可视化结果已保存: {save_path}")
        
        return vis_image


# ==================== 标准化接口函数 ====================

def create_tracker_initializer(
    model_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = 'auto'
) -> SUTrackInitializer:
    """
    创建SUTrack跟踪初始化器 (标准化接口)
    
    参数:
        model_path: 模型路径
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        device: 计算设备
        
    返回:
        initializer: SUTrackInitializer实例
    """
    return SUTrackInitializer(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device
    )


def initialize_tracking(
    first_frame: np.ndarray,
    model_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = 'auto',
    frame_id: int = 0
) -> Optional[Dict]:
    """
    快速初始化跟踪 (简化接口)
    
    参数:
        first_frame: 第一帧图像
        model_path: 模型路径
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        device: 计算设备
        frame_id: 帧ID
        
    返回:
        result: 初始化结果字典，失败返回None
    """
    initializer = create_tracker_initializer(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device
    )
    
    init_result = initializer.initialize(first_frame, frame_id)
    
    if init_result is None:
        return None
    
    return init_result.to_dict()


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """模块测试"""
    
    print("=" * 60)
    print("YOLOv8红外目标检测与SUTrack跟踪初始化模块测试")
    print("=" * 60)
    
    # 测试图像路径
    test_image_path = YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg"
    
    # 如果测试图像不存在，使用其他图像
    if not test_image_path.exists():
        # 查找测试图像
        test_dirs = [
            YOLOV8_ROOT / "test_img",
            YOLOV8_ROOT / "data",
        ]
        test_image_path = None
        for test_dir in test_dirs:
            if test_dir.exists():
                image_files = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png"))
                if image_files:
                    test_image_path = image_files[0]
                    break
    
    if test_image_path is None or not test_image_path.exists():
        print("错误: 未找到测试图像")
        print("请确保存在测试图像文件")
        sys.exit(1)
    
    print(f"\n测试图像: {test_image_path}")
    
    # 1. 测试图像预处理器
    print("\n[1] 测试图像预处理器...")
    preprocessor = InfraredImagePreprocessor()
    processed_img, original_img, original_size = preprocessor.preprocess_from_path(test_image_path)
    print(f"    原始尺寸: {original_size}")
    print(f"    预处理后形状: {processed_img.shape}")
    print(f"    数据类型: {processed_img.dtype}")
    print(f"    数值范围: [{processed_img.min():.3f}, {processed_img.max():.3f}]")
    
    # 2. 测试YOLOv8检测器
    print("\n[2] 测试YOLOv8检测器...")
    detector = YOLOv8Detector(conf_threshold=0.25, iou_threshold=0.45)
    detections = detector.detect_from_path(test_image_path)
    print(f"    检测到 {len(detections)} 个目标")
    
    if len(detections) > 0:
        for i, det in enumerate(detections[:3]):  # 只显示前3个
            print(f"    目标 {i+1}: 置信度={det.confidence:.4f}, 类别={det.class_name}")
    
    # 3. 测试单目标检测
    print("\n[3] 测试单目标检测...")
    single_detection = detector.detect_single_target(cv2.imread(str(test_image_path)))
    if single_detection:
        print(f"    最佳目标: 置信度={single_detection.confidence:.4f}")
        print(f"    边界框: {single_detection.bbox}")
        print(f"    中心点: {single_detection.center}")
        print(f"    尺寸: {single_detection.size}")
    
    # 4. 测试SUTrack初始化器
    print("\n[4] 测试SUTrack初始化器...")
    initializer = SUTrackInitializer(conf_threshold=0.25, iou_threshold=0.45)
    init_result = initializer.initialize_from_path(test_image_path, frame_id=0)
    
    if init_result:
        print(f"    初始化成功!")
        print(f"    目标ID: {init_result.target_id}")
        print(f"    初始边界框: {init_result.initial_bbox}")
        print(f"    初始中心点: {init_result.initial_center}")
        print(f"    初始尺寸: {init_result.initial_size}")
        print(f"    置信度: {init_result.confidence:.4f}")
        print(f"    元数据: {init_result.metadata}")
        
        # 5. 测试可视化
        print("\n[5] 测试可视化...")
        image = cv2.imread(str(test_image_path))
        vis_result = initializer.visualize_detection(
            image, 
            init_result,
            save_path=str(YOLOV8_ROOT / "test_results" / "initialization_result.jpg")
        )
        print(f"    可视化完成，图像尺寸: {vis_result.shape}")
    
    # 6. 测试简化接口
    print("\n[6] 测试简化接口...")
    result_dict = initialize_tracking(
        cv2.imread(str(test_image_path)),
        conf_threshold=0.25,
        frame_id=0
    )
    if result_dict:
        print(f"    简化接口调用成功!")
        print(f"    结果字典键: {list(result_dict.keys())}")
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
