"""
简化版测试脚本 - YOLOv8 红外无人机检测
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# 加载模型 - 使用绝对路径
model_path = r"E:\biyesheji\SUTrack-main11\YOLOv8-main\runs\detect\anti_uav_single_stage16\weights\best.pt"
print(f"加载模型: {model_path}")
model = YOLO(model_path)

# 测试图片 - 使用绝对路径
image_path = r"E:\biyesheji\SUTrack-main11\data\AntI-UAV\yolo_subset\images\test\20190925_111757_1_1_frame000080.jpg"
print(f"测试图片: {image_path}")

# 读取原图
img = cv2.imread(image_path)
img_height, img_width = img.shape[:2]

# 进行检测 - 使用predict模式并保存结果
results = model.predict(image_path, conf=0.25, save=True, project="test_results", name="detection")

print(f"\n检测结果已保存到 test_results/detection/ 目录")
print(f"检测完成！")

print("\n测试完成!")