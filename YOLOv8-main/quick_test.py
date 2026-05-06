"""快速测试YOLOv8模块"""
import sys
from pathlib import Path

YOLOV8_ROOT = Path(__file__).parent
sys.path.insert(0, str(YOLOV8_ROOT))

print("测试YOLOv8模块...")

try:
    from ultralytics import YOLO
    print("✓ YOLO导入成功")
    
    # 加载模型
    model_path = YOLOV8_ROOT / "runs" / "detect" / "anti_uav_single_stage16" / "weights" / "best.pt"
    print(f"加载模型: {model_path}")
    
    model = YOLO(str(model_path))
    print("✓ 模型加载成功")
    
    # 测试获取names属性
    print("检查names属性...")
    if hasattr(model, 'names'):
        print(f"✓ model.names: {model.names}")
    else:
        print("✗ model.names 不存在")
        
    if hasattr(model, 'model') and hasattr(model.model, 'names'):
        print(f"✓ model.model.names: {model.model.names}")
    else:
        print("✗ model.model.names 不存在")
    
    print("\n测试完成!")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
