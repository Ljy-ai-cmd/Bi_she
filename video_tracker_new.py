"""
SUTrack实时检测与跟踪可视化系统
基于YOLOv8目标检测 + SUTrack跟踪算法
功能：
1. YOLOv8自动检测第一帧目标
2. 实时跟踪与可视化
3. 性能监控与统计
4. 多目标轨迹显示
5. 实时FPS和延迟监控
"""

import os
import sys
import cv2
import time
import torch
import numpy as np
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Deque
import threading
import queue

# Add project path
prj = os.path.dirname(os.path.abspath(__file__))
if prj not in sys.path:
    sys.path.insert(0, prj)

# 导入YOLOv8检测模块
YOLOV8_PATH = Path(prj) / "YOLOv8-main"
if str(YOLOV8_PATH) not in sys.path:
    sys.path.insert(0, str(YOLOV8_PATH))

from video_tracker import VideoTracker

# 尝试导入YOLOv8检测器
try:
    from yolo_tracker_initializer import YOLOv8Detector, DetectionResult
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: YOLOv8检测器不可用，将使用传统检测方法")


@dataclass
class PerformanceMetrics:
    """性能监控指标"""
    fps: float = 0.0
    frame_time: float = 0.0
    detection_time: float = 0.0
    tracking_time: float = 0.0
    total_latency: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    
    # 历史记录
    fps_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    latency_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    
    def update(self, frame_time: float, detection_time: float = 0, tracking_time: float = 0):
        """更新性能指标"""
        self.frame_time = frame_time
        self.detection_time = detection_time
        self.tracking_time = tracking_time
        self.total_latency = detection_time + tracking_time
        
        # 计算FPS
        if frame_time > 0:
            self.fps = 1.0 / frame_time
        
        # 更新历史
        self.fps_history.append(self.fps)
        self.latency_history.append(self.total_latency)
    
    def get_avg_fps(self) -> float:
        """获取平均FPS"""
        return np.mean(self.fps_history) if self.fps_history else 0.0
    
    def get_avg_latency(self) -> float:
        """获取平均延迟"""
        return np.mean(self.latency_history) if self.latency_history else 0.0


@dataclass
class TrackTrajectory:
    """跟踪轨迹数据结构"""
    positions: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=100))
    scores: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    def add_point(self, center: Tuple[int, int], score: float, timestamp: float):
        """添加轨迹点"""
        self.positions.append(center)
        self.scores.append(score)
        self.timestamps.append(timestamp)
    
    def get_trajectory_points(self, max_points: int = 50) -> List[Tuple[int, int]]:
        """获取轨迹点列表"""
        return list(self.positions)[-max_points:]


class RealtimeDetectionTracker:
    """
    实时检测与跟踪可视化系统
    
    功能特性：
    - YOLOv8自动目标检测初始化
    - 实时跟踪与可视化
    - 性能监控面板
    - 目标轨迹显示
    - FPS和延迟监控
    """
    
    def __init__(self, 
                 model_path: str,
                 yolo_weight_path: Optional[str] = None,
                 config: str = 'sutrack_b224',
                 enhance_infrared: bool = True,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = 'auto'):
        """
        初始化实时检测跟踪系统
        
        Args:
            model_path: SUTrack模型路径
            yolo_weight_path: YOLOv8权重路径 (可选)
            config: SUTrack配置名称
            enhance_infrared: 是否增强红外图像
            conf_threshold: YOLO检测置信度阈值
            iou_threshold: YOLO NMS IOU阈值
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        # 初始化SUTrack跟踪器
        self.tracker = VideoTracker(
            model_path=model_path,
            yaml_name=config,
            dataset_name='GOT10K',
            enhance_infrared=enhance_infrared
        )
        
        # 初始化YOLOv8检测器
        self.yolo_detector = None
        self.yolo_available = False
        
        if YOLO_AVAILABLE and yolo_weight_path:
            try:
                self.yolo_detector = YOLOv8Detector(
                    model_path=yolo_weight_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    device=device
                )
                self.yolo_available = True
                print(f"✓ YOLOv8检测器初始化成功: {yolo_weight_path}")
            except Exception as e:
                print(f"✗ YOLOv8检测器初始化失败: {e}")
        
        # 性能监控
        self.metrics = PerformanceMetrics()
        
        # 轨迹记录
        self.trajectory = TrackTrajectory()
        
        # 检测结果
        self.detection_results = []
        self.tracking_results = []
        
        # 可视化配置
        self.colors = {
            'primary': (0, 255, 0),      # 绿色 - 跟踪正常
            'warning': (0, 255, 255),     # 黄色 - 跟踪一般
            'danger': (0, 0, 255),        # 红色 - 目标丢失
            'info': (255, 255, 255),      # 白色 - 信息
            'trajectory': (255, 0, 0),    # 蓝色 - 轨迹
            'detection': (0, 165, 255),   # 橙色 - 检测框
        }
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'lost_frames': 0,
            'detection_count': 0,
            'start_time': None,
        }
        
        # 显示配置
        self.show_trajectory = False  # 禁用轨迹显示
        self.show_metrics = True
        self.show_detection_box = True
        
    def detect_with_yolo(self, frame: np.ndarray) -> Optional[Tuple[List[int], float]]:
        """
        使用YOLOv8检测第一帧目标
        
        Args:
            frame: 输入图像
            
        Returns:
            (bbox, confidence) 或 None
            bbox格式: [x, y, w, h]
        """
        if not self.yolo_available or self.yolo_detector is None:
            return None
        
        try:
            # 执行YOLO检测
            detections = self.yolo_detector.detect(frame)
            
            if not detections:
                print("YOLO未检测到目标")
                return None
            
            # 获取最佳检测结果
            best_detection = detections[0]
            
            # 转换bbox格式 [x1, y1, x2, y2] -> [x, y, w, h]
            x1, y1, x2, y2 = best_detection.bbox
            w = x2 - x1
            h = y2 - y1
            
            bbox = [int(x1), int(y1), int(w), int(h)]
            confidence = best_detection.confidence
            
            print(f"YOLO检测到目标: bbox={bbox}, 置信度={confidence:.3f}, 类别={best_detection.class_name}")
            
            return bbox, confidence
            
        except Exception as e:
            print(f"YOLO检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def initialize_first_frame(self,
                              first_frame: np.ndarray,
                              detection_mode: str = 'auto') -> Optional[List[int]]:
        """
        初始化第一帧

        Args:
            first_frame: 第一帧图像
            detection_mode: 'auto'(YOLO自动) / 'manual'(手动选择)

        Returns:
            初始边界框 [x, y, w, h] 或 None
        """
        print("\n" + "="*60)
        print("第一帧目标检测初始化")
        print("="*60)

        init_bbox = None
        confidence = 0.0

        if detection_mode == 'auto' and self.yolo_available:
            # 使用YOLOv8检测
            print("使用YOLOv8进行目标检测...")
            result = self.detect_with_yolo(first_frame)

            if result:
                init_bbox, confidence = result
                self.stats['detection_count'] = 1
            else:
                print("YOLO检测失败，请使用手动模式")

        else:  # manual
            # 手动选择 - 使用命令行输入
            print("请手动输入目标区域坐标...")
            init_bbox = self.select_roi_command_line(first_frame)
            confidence = 1.0
        
        if init_bbox is None:
            print("✗ 目标检测失败")
            return None
        
        # 保存检测结果
        self.detection_results.append({
            'frame_id': 0,
            'bbox': init_bbox,
            'confidence': confidence,
            'mode': detection_mode
        })
        
        print(f"\n✓ 初始化成功")
        print(f"  边界框: {init_bbox}")
        print(f"  置信度: {confidence:.3f}")
        
        # 初始化SUTrack跟踪器
        self.tracker.initialize(first_frame, init_bbox)
        
        # 初始化轨迹
        cx = init_bbox[0] + init_bbox[2] // 2
        cy = init_bbox[1] + init_bbox[3] // 2
        self.trajectory.add_point((cx, cy), confidence, time.time())
        
        return init_bbox
    
    def select_roi_command_line(self, frame: np.ndarray) -> Optional[List[int]]:
        """
        通过命令行输入选择ROI区域（替代GUI选择）
        
        Args:
            frame: 第一帧图像
            
        Returns:
            边界框 [x, y, w, h] 或 None
        """
        h, w = frame.shape[:2]
        print(f"\n图像尺寸: {w}x{h}")
        print("请输入目标区域的坐标（左上角x, 左上角y, 宽度, 高度）")
        print("示例: 100 150 80 60")
        print("或输入 'center' 选择图像中心区域")
        print("或输入 'cancel' 取消")
        
        while True:
            try:
                user_input = input("\n请输入坐标: ").strip().lower()
                
                if user_input == 'cancel':
                    return None
                
                if user_input == 'center':
                    # 默认选择中心区域
                    cx, cy = w // 2, h // 2
                    bw, bh = min(100, w // 4), min(100, h // 4)
                    x = max(0, cx - bw // 2)
                    y = max(0, cy - bh // 2)
                    bbox = [x, y, bw, bh]
                    print(f"选择中心区域: {bbox}")
                    return bbox
                
                # 解析坐标
                coords = list(map(int, user_input.split()))
                if len(coords) != 4:
                    print("错误: 需要输入4个数值（x y w h）")
                    continue
                
                x, y, bw, bh = coords
                
                # 验证坐标
                if x < 0 or y < 0 or bw <= 0 or bh <= 0:
                    print("错误: 坐标必须为正数")
                    continue
                
                if x >= w or y >= h:
                    print(f"错误: 坐标超出图像范围 (图像尺寸: {w}x{h})")
                    continue
                
                if x + bw > w or y + bh > h:
                    print(f"警告: 边界框超出图像范围，将自动裁剪")
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                
                bbox = [x, y, bw, bh]
                print(f"选择的边界框: {bbox}")
                
                # 显示预览（保存到文件）
                preview = frame.copy()
                cv2.rectangle(preview, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                preview_path = "roi_preview.jpg"
                cv2.imwrite(preview_path, preview)
                print(f"预览已保存到: {preview_path}")
                
                # 确认
                confirm = input("确认选择? (y/n): ").strip().lower()
                if confirm == 'y' or confirm == '':
                    return bbox
                
            except ValueError:
                print("错误: 请输入有效的整数坐标")
            except Exception as e:
                print(f"错误: {e}")
    
    def draw_trajectory(self, frame: np.ndarray, color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        绘制目标轨迹
        
        Args:
            frame: 输入帧
            color: 轨迹颜色
            
        Returns:
            绘制后的帧
        """
        if not self.show_trajectory:
            return frame
        
        color = color or self.colors['trajectory']
        points = self.trajectory.get_trajectory_points(max_points=50)
        
        if len(points) < 2:
            return frame
        
        # 绘制轨迹线
        for i in range(1, len(points)):
            # 根据时间衰减透明度
            alpha = int(255 * (i / len(points)))
            thickness = max(1, int(3 * (i / len(points))))
            
            pt1 = points[i-1]
            pt2 = points[i]
            cv2.line(frame, pt1, pt2, color, thickness)
        
        # 绘制当前位置点
        if points:
            cv2.circle(frame, points[-1], 5, (0, 0, 255), -1)
        
        return frame
    
    def draw_metrics_panel(self, frame: np.ndarray) -> np.ndarray:
        """
        绘制性能监控面板
        
        Args:
            frame: 输入帧
            
        Returns:
            绘制后的帧
        """
        if not self.show_metrics:
            return frame
        
        h, w = frame.shape[:2]
        
        # 面板背景
        panel_height = 120
        panel_width = 280
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_width - 10, 10), 
                     (w - 10, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制边框
        cv2.rectangle(frame, (w - panel_width - 10, 10), 
                     (w - 10, panel_height), self.colors['info'], 1)
        
        # 性能指标文本
        texts = [
            f"FPS: {self.metrics.fps:.1f} (avg: {self.metrics.get_avg_fps():.1f})",
            f"Latency: {self.metrics.total_latency*1000:.1f}ms",
            f"Track Time: {self.metrics.tracking_time*1000:.1f}ms",
            f"Lost: {self.stats['lost_frames']}/{self.stats['total_frames']}",
        ]
        
        y_offset = 30
        for i, text in enumerate(texts):
            # 根据数值选择颜色
            if i == 0 and self.metrics.fps < 20:
                color = self.colors['danger']
            elif i == 1 and self.metrics.total_latency > 0.1:
                color = self.colors['warning']
            else:
                color = self.colors['info']
            
            cv2.putText(frame, text, (w - panel_width, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def draw_detection_box(self, frame: np.ndarray, bbox: List[int], 
                          confidence: float, is_detection: bool = False) -> np.ndarray:
        """
        绘制检测/跟踪框
        
        Args:
            frame: 输入帧
            bbox: 边界框 [x, y, w, h]
            confidence: 置信度
            is_detection: 是否为检测框 (vs 跟踪框)
            
        Returns:
            绘制后的帧
        """
        if not self.show_detection_box:
            return frame
        
        x, y, w, h = [int(v) for v in bbox]
        
        # 选择颜色
        if is_detection:
            color = self.colors['detection']
            label = f"Detect: {confidence:.3f}"
        else:
            if confidence >= 0.5:
                color = self.colors['primary']
            elif confidence >= 0.3:
                color = self.colors['warning']
            else:
                color = self.colors['danger']
            label = f"Track: {confidence:.3f}"
        
        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # 绘制标签背景
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y - 10, label_size[1] + 10)
        cv2.rectangle(frame, (x, label_y - label_size[1] - 5),
                     (x + label_size[0], label_y + 5), color, -1)
        
        # 绘制标签文字
        cv2.putText(frame, label, (x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict]:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            frame_id: 帧ID
            
        Returns:
            (可视化帧, 结果字典)
        """
        start_time = time.time()
        
        # 跟踪
        track_start = time.time()
        outputs = self.tracker.track(frame)
        track_time = time.time() - track_start
        
        bbox = outputs['target_bbox']
        score = outputs['best_score']
        score_val = score.item() if hasattr(score, 'item') else float(score)
        
        # 更新统计
        self.stats['total_frames'] += 1
        if score_val < 0.3:
            self.stats['lost_frames'] += 1
        
        # 更新轨迹
        cx = int(bbox[0] + bbox[2] / 2)
        cy = int(bbox[1] + bbox[3] / 2)
        self.trajectory.add_point((cx, cy), score_val, time.time())
        
        # 保存结果
        result = {
            'frame_id': frame_id,
            'bbox': bbox,
            'score': score_val,
            'timestamp': time.time()
        }
        self.tracking_results.append(result)
        
        # 可视化
        vis_frame = frame.copy()
        
        # 绘制跟踪框
        vis_frame = self.draw_detection_box(vis_frame, bbox, score_val, is_detection=False)
        
        # 绘制性能面板
        frame_time = time.time() - start_time
        self.metrics.update(frame_time, detection_time=0, tracking_time=track_time)
        vis_frame = self.draw_metrics_panel(vis_frame)
        
        # 添加帧信息
        info_text = f"Frame: {frame_id}"
        if score_val < 0.3:
            info_text += " | LOST"
        cv2.putText(vis_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['info'], 2)
        
        return vis_frame, result
    
    def run(self, 
           video_source: str,
           output_dir: str = 'tracking_results',
           detection_mode: str = 'auto',
           display: bool = True,
           save_video: bool = True,
           max_frames: Optional[int] = None) -> Dict:
        """
        运行实时检测跟踪
        
        Args:
            video_source: 视频路径或摄像头ID (0, 1, ...)
            output_dir: 输出目录
            detection_mode: 检测模式 ('auto', 'traditional', 'manual')
            display: 是否显示实时画面
            save_video: 是否保存结果视频
            max_frames: 最大处理帧数 (None表示处理全部)
            
        Returns:
            运行统计信息
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频源
        if video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
            source_name = f"camera_{video_source}"
        else:
            cap = cv2.VideoCapture(video_source)
            source_name = Path(video_source).stem
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {video_source}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n视频信息: {width}x{height} @ {fps}fps")
        if total_frames > 0:
            print(f"总帧数: {total_frames}")
        
        # 读取第一帧
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("无法读取第一帧")
        
        # 第一帧目标检测初始化
        init_bbox = self.initialize_first_frame(first_frame, detection_mode)
        
        if init_bbox is None:
            print("✗ 初始化失败，退出")
            cap.release()
            return {}
        
        # 设置视频写入
        writer = None
        if save_video:
            output_video_path = os.path.join(output_dir, f'{source_name}_tracked.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"\n输出视频: {output_video_path}")
        
        # 初始化统计
        self.stats['start_time'] = time.time()
        
        print("\n" + "="*60)
        print("开始实时跟踪")
        print("="*60)
        print("快捷键:")
        print("  'q' - 退出")
        print("  ' ' (空格) - 暂停/继续")
        print("  'm' - 切换性能面板")
        print("="*60 + "\n")
        
        frame_id = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 处理帧
                    vis_frame, result = self.process_frame(frame, frame_id)
                    
                    # 写入视频
                    if writer:
                        writer.write(vis_frame)
                    
                    frame_id += 1
                    
                    # 检查最大帧数
                    if max_frames and frame_id >= max_frames:
                        print(f"达到最大帧数限制: {max_frames}")
                        break
                    
                    # 打印进度
                    if frame_id % 30 == 0:
                        avg_fps = self.metrics.get_avg_fps()
                        avg_latency = self.metrics.get_avg_latency() * 1000
                        print(f"Frame {frame_id} | FPS: {avg_fps:.1f} | "
                              f"Latency: {avg_latency:.1f}ms | "
                              f"Score: {result['score']:.3f}")
                
                # 显示
                if display:
                    try:
                        cv2.imshow('SUTrack Realtime Detection', vis_frame)
                        
                        key = cv2.waitKey(1 if not paused else 0) & 0xFF
                        
                        if key == ord('q'):
                            print("用户退出")
                            break
                        elif key == ord(' '):
                            paused = not paused
                            print("暂停" if paused else "继续")
                        elif key == ord('m'):
                            self.show_metrics = not self.show_metrics
                            print(f"性能面板: {'开启' if self.show_metrics else '关闭'}")
                    except cv2.error as e:
                        # OpenCV显示不可用（无头模式）
                        if "not implemented" in str(e):
                            print("警告: OpenCV显示功能不可用，切换到无头模式")
                            display = False
                        else:
                            raise
        
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            # 释放资源
            cap.release()
            if writer:
                writer.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # 忽略OpenCV显示错误
            
            # 计算最终统计
            elapsed_time = time.time() - self.stats['start_time']
            avg_fps = frame_id / elapsed_time if elapsed_time > 0 else 0
            
            final_stats = {
                'total_frames': frame_id,
                'elapsed_time': elapsed_time,
                'avg_fps': avg_fps,
                'lost_frames': self.stats['lost_frames'],
                'lost_rate': self.stats['lost_frames'] / frame_id if frame_id > 0 else 0,
                'detection_count': self.stats['detection_count'],
            }
            
            # 保存结果
            results_path = os.path.join(output_dir, f'{source_name}_results.json')
            with open(results_path, 'w') as f:
                json.dump({
                    'stats': final_stats,
                    'tracking_results': self.tracking_results,
                    'detection_results': self.detection_results,
                }, f, indent=2)
            
            print("\n" + "="*60)
            print("跟踪完成")
            print("="*60)
            print(f"总帧数: {final_stats['total_frames']}")
            print(f"运行时间: {final_stats['elapsed_time']:.2f}s")
            print(f"平均FPS: {final_stats['avg_fps']:.2f}")
            print(f"丢失帧数: {final_stats['lost_frames']} ({final_stats['lost_rate']*100:.1f}%)")
            print(f"结果保存: {results_path}")
            print("="*60)
            
            return final_stats


def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SUTrack实时检测与跟踪')
    parser.add_argument('--video', type=str, required=True, help='视频路径或摄像头ID')
    parser.add_argument('--model', type=str, 
                       default=r'checkpoints\train\sutrack\sutrack_b224\SUTRACK_ep0180.pth.tar',
                       help='SUTrack模型路径')
    parser.add_argument('--yolo-weight', type=str,
                       default='YOLOv8-main/runs/detect/anti_uav_single_stage16/weights/best.pt',
                       help='YOLOv8权重路径')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'manual'],
                       help='检测模式')
    parser.add_argument('--output', type=str, default='tracking_results',
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='YOLO检测置信度阈值')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='计算设备')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示实时画面')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果视频')
    
    args = parser.parse_args()
    
    # 创建跟踪器
    tracker = RealtimeDetectionTracker(
        model_path=args.model,
        yolo_weight_path=args.yolo_weight if args.mode == 'auto' else None,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # 运行跟踪
    stats = tracker.run(
        video_source=args.video,
        output_dir=args.output,
        detection_mode=args.mode,
        display=not args.no_display,
        save_video=not args.no_save
    )
    
    return stats


if __name__ == '__main__':
    main()
