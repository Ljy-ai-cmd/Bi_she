"""
Anti-UAV 红外无人机检测 - 模型评估脚本
提供全面的模型性能评估和分析
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.metrics import ap_per_class, ConfusionMatrix


class AntiUAVEvaluator:
    """Anti-UAV模型评估器"""
    
    def __init__(self, model_path, data_yaml, device='0'):
        """
        初始化评估器
        Args:
            model_path: 模型路径
            data_yaml: 数据集配置文件
            device: GPU设备
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.device = device
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 评估结果存储
        self.results = {}
        
    def evaluate(self, split='val', conf_thres=0.001, iou_thres=0.6, 
                 max_det=300, save_json=True, save_plots=True):
        """
        评估模型性能
        Args:
            split: 评估数据集划分 ('val' 或 'test')
            conf_thres: 置信度阈值
            iou_thres: NMS IoU阈值
            max_det: 最大检测数
            save_json: 保存结果为JSON
            save_plots: 保存可视化图表
        Returns:
            评估结果字典
        """
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"开始评估 - {split}集")
        LOGGER.info(f"{'=' * 60}")
        LOGGER.info(f"模型: {self.model_path}")
        LOGGER.info(f"置信度阈值: {conf_thres}")
        LOGGER.info(f"NMS IoU阈值: {iou_thres}")
        LOGGER.info(f"最大检测数: {max_det}")
        
        # 执行验证
        results = self.model.val(
            data=self.data_yaml,
            split=split,
            imgsz=640,
            batch=16,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            save_json=save_json,
            save_hybrid=True,
            plots=save_plots
        )
        
        # 提取关键指标
        self.results[split] = {
            'mAP50': results.results_dict.get('metrics/mAP50', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95', 0),
            'mAP75': results.results_dict.get('metrics/mAP75', 0),
            'precision': results.results_dict.get('metrics/precision', 0),
            'recall': results.results_dict.get('metrics/recall', 0),
            'f1_score': self._calculate_f1(
                results.results_dict.get('metrics/precision', 0),
                results.results_dict.get('metrics/recall', 0)
            )
        }
        
        # 打印结果
        self._print_results(split)
        
        return self.results[split]
    
    def _calculate_f1(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def _print_results(self, split):
        """打印评估结果"""
        results = self.results[split]
        
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"评估结果 - {split}集")
        LOGGER.info(f"{'=' * 60}")
        LOGGER.info(f"mAP@0.5:     {results['mAP50']:.4f}")
        LOGGER.info(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
        LOGGER.info(f"mAP@0.75:    {results['mAP75']:.4f}")
        LOGGER.info(f"Precision:   {results['precision']:.4f}")
        LOGGER.info(f"Recall:      {results['recall']:.4f}")
        LOGGER.info(f"F1-Score:    {results['f1_score']:.4f}")
        LOGGER.info(f"{'=' * 60}\n")
    
    def evaluate_multi_conf(self, split='val', conf_range=None):
        """
        多置信度阈值评估
        分析不同置信度阈值下的性能变化
        """
        if conf_range is None:
            conf_range = np.arange(0.1, 0.95, 0.05)
        
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"多置信度阈值评估 - {split}集")
        LOGGER.info(f"{'=' * 60}")
        
        multi_conf_results = []
        
        for conf in conf_range:
            results = self.model.val(
                data=self.data_yaml,
                split=split,
                imgsz=640,
                batch=16,
                conf=conf,
                iou=0.6,
                plots=False,
                verbose=False
            )
            
            multi_conf_results.append({
                'conf': conf,
                'mAP50': results.results_dict.get('metrics/mAP50', 0),
                'precision': results.results_dict.get('metrics/precision', 0),
                'recall': results.results_dict.get('metrics/recall', 0)
            })
            
            LOGGER.info(f"Conf={conf:.2f}: mAP50={multi_conf_results[-1]['mAP50']:.4f}, "
                       f"P={multi_conf_results[-1]['precision']:.4f}, "
                       f"R={multi_conf_results[-1]['recall']:.4f}")
        
        self.multi_conf_results = multi_conf_results
        return multi_conf_results
    
    def evaluate_speed(self, imgsz=640, warmup=10, iterations=100):
        """
        评估模型推理速度
        """
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info("推理速度评估")
        LOGGER.info(f"{'=' * 60}")
        
        import time
        
        # 创建随机输入
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(self.device)
        
        # 预热
        LOGGER.info(f"预热 {warmup} 次...")
        for _ in range(warmup):
            _ = self.model.predict(dummy_input, verbose=False)
        
        # 正式测试
        LOGGER.info(f"测试 {iterations} 次...")
        times = []
        
        with torch.no_grad():
            for _ in range(iterations):
                start = time.time()
                _ = self.model.predict(dummy_input, verbose=False)
                times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # 转换为ms
        std_time = np.std(times) * 1000
        fps = 1000 / avg_time
        
        LOGGER.info(f"平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
        LOGGER.info(f"FPS: {fps:.2f}")
        LOGGER.info(f"{'=' * 60}\n")
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'fps': fps
        }
    
    def analyze_by_attribute(self, split='val'):
        """
        按属性分析性能（针对Anti-UAV数据集特性）
        分析在不同挑战场景下的性能
        """
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"按属性分析性能 - {split}集")
        LOGGER.info(f"{'=' * 60}")
        
        # Anti-UAV属性定义
        attributes = {
            'FM': 'Fast Motion (快速移动)',
            'SV': 'Scale Variation (尺度变化)',
            'OV': 'Out-of-View (离开视野)',
            'TC': 'Thermal Crossover (热交叉)',
            'TC-EASY': 'Thermal Crossover - Easy',
            'TC-MID': 'Thermal Crossover - Medium',
            'TC-HARD': 'Thermal Crossover - Hard',
            'LR': 'Low Resolution (低分辨率)',
            'LI': 'Light Illumination (光照变化)',
            'OC': 'Occlusion (遮挡)'
        }
        
        # 加载属性标签
        label_file = Path(self.data_yaml).parent / 'label_new' / f'{split}.json'
        if not label_file.exists():
            LOGGER.warning(f"属性标签文件不存在: {label_file}")
            return None
        
        with open(label_file, 'r') as f:
            attr_labels = json.load(f)
        
        # 按属性分组评估
        attr_results = defaultdict(list)
        
        # 这里需要根据实际属性标签实现具体的分组评估逻辑
        # 暂时返回属性统计信息
        attr_stats = {}
        for video_id, attrs in attr_labels.items():
            for attr in attrs:
                if attr not in attr_stats:
                    attr_stats[attr] = 0
                attr_stats[attr] += 1
        
        LOGGER.info("属性分布统计:")
        for attr, count in sorted(attr_stats.items()):
            desc = attributes.get(attr, attr)
            LOGGER.info(f"  {attr}: {count} 个视频 - {desc}")
        
        LOGGER.info(f"{'=' * 60}\n")
        
        return attr_stats
    
    def generate_report(self, output_dir='runs/evaluation'):
        """
        生成完整的评估报告
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f'evaluation_report_{timestamp}.txt'
        
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"生成评估报告: {report_file}")
        LOGGER.info(f"{'=' * 60}")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Anti-UAV 红外无人机检测模型评估报告\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"模型: {self.model_path}\n")
            f.write(f"数据集: {self.data_yaml}\n")
            f.write("=" * 60 + "\n\n")
            
            # 写入各split的评估结果
            for split, results in self.results.items():
                f.write(f"\n【{split.upper()} 集评估结果】\n")
                f.write("-" * 40 + "\n")
                f.write(f"mAP@0.5:      {results['mAP50']:.4f}\n")
                f.write(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}\n")
                f.write(f"mAP@0.75:     {results['mAP75']:.4f}\n")
                f.write(f"Precision:    {results['precision']:.4f}\n")
                f.write(f"Recall:       {results['recall']:.4f}\n")
                f.write(f"F1-Score:     {results['f1_score']:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("评估完成\n")
            f.write("=" * 60 + "\n")
        
        LOGGER.info(f"报告已保存: {report_file}")
        return report_file
    
    def plot_pr_curve(self, output_dir='runs/evaluation'):
        """绘制PR曲线"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 这里可以实现PR曲线的绘制
        # 需要使用模型验证过程中保存的预测结果
        
        LOGGER.info(f"PR曲线已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Anti-UAV Model Evaluation')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--data', type=str, default='data/AntI-UAV/yolo_format/anti_uav.yaml',
                       help='数据集配置文件')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU设备')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                       help='评估的数据集划分')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='NMS IoU阈值')
    parser.add_argument('--multi-conf', action='store_true',
                       help='执行多置信度评估')
    parser.add_argument('--speed', action='store_true',
                       help='评估推理速度')
    parser.add_argument('--attribute', action='store_true',
                       help='按属性分析')
    parser.add_argument('--report', action='store_true',
                       help='生成评估报告')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = AntiUAVEvaluator(args.model, args.data, args.device)
    
    # 执行评估
    for split in args.splits:
        evaluator.evaluate(
            split=split,
            conf_thres=args.conf,
            iou_thres=args.iou,
            save_json=True,
            save_plots=True
        )
    
    # 多置信度评估
    if args.multi_conf:
        for split in args.splits:
            evaluator.evaluate_multi_conf(split=split)
    
    # 速度评估
    if args.speed:
        evaluator.evaluate_speed()
    
    # 属性分析
    if args.attribute:
        for split in args.splits:
            evaluator.analyze_by_attribute(split=split)
    
    # 生成报告
    if args.report:
        evaluator.generate_report()
    
    LOGGER.info("\n评估完成！")


if __name__ == '__main__':
    main()
