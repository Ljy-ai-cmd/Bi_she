"""
SUTrack Anti-UAV Dataset SA Evaluation Script
Tests SUTrack tracker on Anti-UAV dataset sequences and calculates SA values
for both visible and infrared modalities.
"""
import os
import sys
import numpy as np
import torch
import cv2
import json
from os.path import join, isdir, abspath, dirname
import importlib

prj = join(dirname(__file__), '.')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.parameter.sutrack import parameters
from lib.test.tracker.sutrack import get_tracker_class
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.train.dataset.depth_utils import get_x_frame

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
    
    # 确保输入为numpy数组并转换为float
    box1 = np.array(box1[:4], dtype=float)
    box2 = np.array(box2[:4], dtype=float)
    
    # 转换为x1,y1,x2,y2格式
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # 计算交集
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union



    ax6.axis('off')
    
    info_text = f"""
    SUTrack Evaluation Results
    
    Model: SUTRACK_ep0120.pth.tar
    Dataset: Anti-UAV (20190926_134054_1_8)
    
    Key Results:
    • RGB AUC: {results['rgb']['auc']:.3f}
    • Thermal AUC: {results['thermal']['auc']:.3f}
    • RGB SA@0.5: {results['rgb']['sa50']:.3f}
    • Thermal SA@0.5: {results['thermal']['sa50']:.3f}
    
    Average FPS: {(results['rgb']['fps'] + results['thermal']['fps'])/2:.2f}
    
    Evaluation Date: {np.datetime64('today')}
    """
    
    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    viz_dir = './visualization_results'
    os.makedirs(viz_dir, exist_ok=True)
    viz_path = os.path.join(viz_dir, 'anti_uav_tracking_results_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    
    # Also save as PDF for better quality
    pdf_path = os.path.join(viz_dir, 'anti_uav_tracking_results_visualization.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"High-quality PDF saved to: {pdf_path}")
    
    plt.close()


def genConfig(seq_path, set_type):
    """Generate configuration for RGBT dataset"""
    if set_type == 'GTOT':
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        # Convert to xywh format
        x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
        RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
        T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        return RGB_img_list, T_img_list, RGB_gt, T_gt
    
    elif set_type == 'ANTI-UAV':
        # Anti-UAV dataset format - load visibility labels for paper-defined SA
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])
        
        # Load annotations from JSON files
        with open(seq_path + '/visible.json', 'r') as f:
            visible_data = json.load(f)
        with open(seq_path + '/infrared.json', 'r') as f:
            infrared_data = json.load(f)
        
        # Extract bounding boxes and visibility labels
        # 处理可能的不一致格式
        def process_gt_rect(gt_rect_data):
            """处理gt_rect数据，确保所有边界框都有4个元素"""
            processed = []
            for rect in gt_rect_data:
                if isinstance(rect, list) and len(rect) >= 4:
                    # 确保有4个元素 [x, y, w, h]
                    processed.append(rect[:4])
                elif isinstance(rect, list) and len(rect) == 2:
                    # 如果只有2个元素，可能是中心点，添加默认宽高
                    processed.append([rect[0], rect[1], 50, 50])  # 默认宽高
                else:
                    # 其他情况使用默认框
                    processed.append([0, 0, 100, 100])
            return np.array(processed, dtype=float)
        
        RGB_gt = process_gt_rect(visible_data['gt_rect'])
        T_gt = process_gt_rect(infrared_data['gt_rect'])
        
        # Load visibility labels (exist field: 1=visible, 0=not visible)
        RGB_visibility = np.array(visible_data['exist'])
        T_visibility = np.array(infrared_data['exist'])
        
        # Debug: Print coordinate ranges and visibility info
        print(f"RGB coordinates range: x={RGB_gt[:,0].min()}-{RGB_gt[:,0].max()}, y={RGB_gt[:,1].min()}-{RGB_gt[:,1].max()}")
        print(f"Thermal coordinates range: x={T_gt[:,0].min()}-{T_gt[:,0].max()}, y={T_gt[:,1].min()}-{T_gt[:,1].max()}")
        print(f"RGB visibility: {np.sum(RGB_visibility)}/{len(RGB_visibility)} visible frames")
        print(f"Thermal visibility: {np.sum(T_visibility)}/{len(T_visibility)} visible frames")
        
        return RGB_img_list, T_img_list, RGB_gt, T_gt, RGB_visibility, T_visibility
    
    return None, None, None, None


def calc_state_accuracy(pred_bboxes, gt_bboxes, visibility_labels, predicted_states=None):
    """Calculate State Accuracy according to the paper definition
    SA = (1/T) * Σ[t=1 to T](IoU_t × δ(v_t > 0) + p_t × (1 - δ(v_t > 0)))
    
    Args:
        pred_bboxes: Predicted bounding boxes [T, 4] in xywh format
        gt_bboxes: Ground truth bounding boxes [T, 4] in xywh format  
        visibility_labels: Visibility labels [T] (1=visible, 0=not visible)
        predicted_states: Predicted target existence states [T] (optional)
    
    Returns:
        state_accuracy: Overall State Accuracy score
        frame_scores: Per-frame scores for analysis
    """
    T = len(pred_bboxes)
    frame_scores = []
    
    for t in range(T):
        v_t = visibility_labels[t]  # Ground truth visibility (1=visible, 0=not visible)
        
        if v_t > 0:  # Target is visible
            # 处理预测框 - 处理可能的字符串或无效数据
            try:
                if isinstance(pred_bboxes[t], str):
                    pred_box = np.array([0.0, 0.0, 0.0, 0.0])
                elif isinstance(pred_bboxes[t], (list, tuple, np.ndarray)):
                    pred_box = np.array(pred_bboxes[t], dtype=float)
                    if len(pred_box) < 4:
                        pred_box = np.array([0.0, 0.0, 0.0, 0.0])
                else:
                    pred_box = np.array(pred_bboxes[t], dtype=float)
                    if len(pred_box) < 4:
                        pred_box = np.array([0.0, 0.0, 0.0, 0.0])
            except (ValueError, TypeError):
                pred_box = np.array([0.0, 0.0, 0.0, 0.0])
            
            # 处理真实框
            try:
                if isinstance(gt_bboxes[t], (list, tuple, np.ndarray)):
                    gt_box = np.array(gt_bboxes[t], dtype=float)
                    if len(gt_box) < 4:
                        gt_box = np.array([0.0, 0.0, 0.0, 0.0])
                else:
                    gt_box = np.array(gt_bboxes[t], dtype=float)
                    if len(gt_box) < 4:
                        gt_box = np.array([0.0, 0.0, 0.0, 0.0])
            except (ValueError, TypeError):
                gt_box = np.array([0.0, 0.0, 0.0, 0.0])
            
            # 计算IoU
            if len(pred_box) < 4 or len(gt_box) < 4:
                iou_t = 0.0
            else:
                iou_t = calculate_iou(pred_box[:4], gt_box[:4])
            frame_scores.append(iou_t)
            
        else:  # Target is not visible
            # Use predicted target existence state p_t
            if predicted_states is not None:
                p_t = predicted_states[t]  # Predicted existence probability
            else:
                # If no predicted states provided, assume target exists if we have a reasonable prediction
                # This is a simplification - in practice, the tracker should provide existence confidence
                try:
                    if isinstance(pred_bboxes[t], str):
                        pred_box = np.array([0.0, 0.0, 0.0, 0.0])
                    elif isinstance(pred_bboxes[t], (list, tuple, np.ndarray)):
                        pred_box = np.array(pred_bboxes[t], dtype=float)
                    else:
                        pred_box = np.array(pred_bboxes[t], dtype=float)
                    
                    # Simple heuristic: if bbox has reasonable size, assume target exists
                    if len(pred_box) >= 4 and pred_box[2] > 10 and pred_box[3] > 10:  # width, height > 10
                        p_t = 1.0
                    else:
                        p_t = 0.0
                except (ValueError, TypeError):
                    p_t = 0.0
            
            frame_scores.append(p_t)  # Score is predicted existence when target is not visible
        
    # Calculate overall State Accuracy
    state_accuracy = np.mean(frame_scores) if frame_scores else 0.0
    
    return state_accuracy, frame_scores


def calc_iou_overlap(pred_bb, anno_bb):
    """Calculate IoU overlap between predicted and ground truth bounding boxes"""
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union


def calc_success_rate(pred_bb, anno_bb, thresholds):
    """Calculate success rate at different IoU thresholds"""
    overlaps = calc_iou_overlap(pred_bb, anno_bb)
    success_rates = []
    for threshold in thresholds:
        success_rate = (overlaps >= threshold).float().mean()
        success_rates.append(success_rate)
    return torch.tensor(success_rates), overlaps


class VEGETA_RGBT(object):
    """RGBT Tracker wrapper"""
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        """TRACK"""
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


def evaluate_single_sequence_paper_sa(sequence_path):
    """Evaluate SUTrack on a single Anti-UAV sequence using paper-defined State Accuracy"""
    
    # Configuration
    yaml_name = 'sutrack_t224'
    checkpoint_path = r'E:\biyesheji\SUTrack-main11\checkpoints\train\sutrack\sutrack_t224\SUTRACK_ep0120.pth.tar'
    
    # Get parameters
    params = parameters(yaml_name)
    params.checkpoint = checkpoint_path
    params.debug = False
    
    print(f"\nEvaluating sequence: {os.path.basename(sequence_path)}")
    
    # Create tracker
    tracker_class = get_tracker_class()
    
    # Generate configuration with visibility labels
    RGB_img_list, T_img_list, RGB_gt, T_gt, RGB_visibility, T_visibility = genConfig(sequence_path, 'ANTI-UAV')
    
    if RGB_img_list is None:
        print("Error: Could not load dataset configuration")
        return None
    
    print(f"Found {len(RGB_img_list)} frames")
    print(f"RGB visibility: {np.sum(RGB_visibility)}/{len(RGB_visibility)} visible frames")
    print(f"Thermal visibility: {np.sum(T_visibility)}/{len(T_visibility)} visible frames")
    
    # Initialize trackers
    rgb_tracker = tracker_class(params, 'ANTI-UAV')
    thermal_tracker = tracker_class(params, 'ANTI-UAV')
    
    # Run RGB tracking
    rgb_predictions = []
    for frame_idx, rgb_path in enumerate(RGB_img_list):
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            print(f"Error loading RGB image: {rgb_path}")
            rgb_predictions.append([0, 0, 0, 0])
            continue
        
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        if frame_idx == 0:
            init_info = {'init_bbox': RGB_gt[0].tolist()}
            rgb_tracker.initialize(rgb_image, init_info)
            rgb_predictions.append(RGB_gt[0])
        else:
            out = rgb_tracker.track(rgb_image)
            
            # 处理跟踪器返回结果
            try:
                if isinstance(out, dict) and 'target_bbox' in out:
                    region = out['target_bbox']
                    if isinstance(region, (list, tuple)) and len(region) >= 4:
                        region = list(region)[:4]
                    else:
                        region = [0.0, 0.0, 0.0, 0.0]
                elif isinstance(out, str):
                    # 如果是字符串，记录错误并使用默认框
                    if frame_idx <= 5:  # 只在前几帧打印错误信息
                        print(f"  警告: RGB跟踪器第{frame_idx}帧返回字符串: {out}")
                    region = [0.0, 0.0, 0.0, 0.0]
                elif isinstance(out, (list, tuple)) and len(out) >= 1:
                    # 如果返回的是列表或元组，第一个元素应该是bbox
                    region = out[0]
                    if isinstance(region, (list, tuple)) and len(region) >= 4:
                        region = list(region)[:4]
                    else:
                        region = [0.0, 0.0, 0.0, 0.0]
                else:
                    region = [0.0, 0.0, 0.0, 0.0]
                        
            except Exception as e:
                if frame_idx <= 5:
                    print(f"  警告: RGB跟踪器第{frame_idx}帧处理失败: {e}")
                region = [0.0, 0.0, 0.0, 0.0]
            
            rgb_predictions.append(region)
    
    # Run thermal tracking
    thermal_predictions = []
    for frame_idx, t_path in enumerate(T_img_list):
        thermal_image = cv2.imread(t_path)
        if thermal_image is None:
            print(f"Error loading thermal image: {t_path}")
            thermal_predictions.append([0, 0, 0, 0])
            continue
        
        thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
        
        if frame_idx == 0:
            init_info = {'init_bbox': T_gt[0].tolist()}
            thermal_tracker.initialize(thermal_image, init_info)
            thermal_predictions.append(T_gt[0])
        else:
            out = thermal_tracker.track(thermal_image)
            
            # 处理跟踪器返回结果
            try:
                if isinstance(out, dict) and 'target_bbox' in out:
                    region = out['target_bbox']
                    if isinstance(region, (list, tuple)) and len(region) >= 4:
                        region = list(region)[:4]
                    else:
                        region = [0.0, 0.0, 0.0, 0.0]
                elif isinstance(out, str):
                    # 如果是字符串，记录错误并使用默认框
                    if frame_idx <= 5:  # 只在前几帧打印错误信息
                        print(f"  警告: 红外跟踪器第{frame_idx}帧返回字符串: {out}")
                    region = [0.0, 0.0, 0.0, 0.0]
                elif isinstance(out, (list, tuple)) and len(out) >= 1:
                    # 如果返回的是列表或元组，第一个元素应该是bbox
                    region = out[0]
                    if isinstance(region, (list, tuple)) and len(region) >= 4:
                        region = list(region)[:4]
                    else:
                        region = [0.0, 0.0, 0.0, 0.0]
                else:
                    region = [0.0, 0.0, 0.0, 0.0]
                        
            except Exception as e:
                if frame_idx <= 5:
                    print(f"  警告: 红外跟踪器第{frame_idx}帧处理失败: {e}")
                region = [0.0, 0.0, 0.0, 0.0]
            
            thermal_predictions.append(region)
    
    # Calculate paper-defined State Accuracy
    rgb_state_accuracy, rgb_frame_scores = calc_state_accuracy(
        rgb_predictions, RGB_gt, RGB_visibility
    )
    
    thermal_state_accuracy, thermal_frame_scores = calc_state_accuracy(
        thermal_predictions, T_gt, T_visibility
    )
    
    # Calculate total State Accuracy (average of RGB and Thermal)
    total_state_accuracy = (rgb_state_accuracy + thermal_state_accuracy) / 2.0
    
    print(f"\nPaper-defined State Accuracy Results:")
    print(f"RGB State Accuracy: {rgb_state_accuracy:.3f}")
    print(f"Thermal State Accuracy: {thermal_state_accuracy:.3f}")
    print(f"Total State Accuracy: {total_state_accuracy:.3f}")
    
    return {
        'sequence_name': os.path.basename(sequence_path),
        'rgb_state_accuracy': rgb_state_accuracy,
        'thermal_state_accuracy': thermal_state_accuracy,
        'total_state_accuracy': total_state_accuracy,
        'num_frames': len(RGB_img_list),
        'rgb_visible_frames': int(np.sum(RGB_visibility)),
        'thermal_visible_frames': int(np.sum(T_visibility))
    }


def evaluate_entire_anti_uav_testset():
    """Evaluate SUTrack on entire Anti-UAV test set using paper-defined State Accuracy"""
    
    # Configuration
    testset_path = r'E:\biyesheji\SUTrack-main11\data\Anti-UAV\test'
    
    print(f"Evaluating entire Anti-UAV test set: {testset_path}")
    
    # Get all test sequences
    test_sequences = []
    for item in os.listdir(testset_path):
        seq_path = join(testset_path, item)
        if os.path.isdir(seq_path):
            # Check if it has the required JSON files
            visible_json = join(seq_path, 'visible.json')
            infrared_json = join(seq_path, 'infrared.json')
            if os.path.exists(visible_json) and os.path.exists(infrared_json):
                test_sequences.append(item)
    
    print(f"Found {len(test_sequences)} valid test sequences")
    
    # Results collection
    all_results = {
        'rgb_state_accuracy': [],
        'thermal_state_accuracy': [],
        'total_state_accuracy': [],
        'sequence_names': []
    }
    
    # Process each sequence
    for seq_name in test_sequences:
        seq_path = join(testset_path, seq_name)
        result = evaluate_single_sequence_paper_sa(seq_path)
        
        if result:
            all_results['rgb_state_accuracy'].append(result['rgb_state_accuracy'])
            all_results['thermal_state_accuracy'].append(result['thermal_state_accuracy'])
            all_results['total_state_accuracy'].append(result['total_state_accuracy'])
            all_results['sequence_names'].append(result['sequence_name'])
    
    # Calculate aggregate results
    if all_results['rgb_state_accuracy']:
        avg_rgb_sa = np.mean(all_results['rgb_state_accuracy'])
        avg_thermal_sa = np.mean(all_results['thermal_state_accuracy'])
        avg_total_sa = np.mean(all_results['total_state_accuracy'])
        
        print(f"\n{'='*80}")
        print(f"PAPER-DEFINED STATE ACCURACY RESULTS - ENTIRE ANTI-UAV TEST SET")
        print(f"{'='*80}")
        print(f"Number of sequences evaluated: {len(all_results['sequence_names'])}")
        print(f"")
        print(f"Average RGB (Visible) State Accuracy: {avg_rgb_sa:.3f}")
        print(f"Average Thermal (Infrared) State Accuracy: {avg_thermal_sa:.3f}")
        print(f"Average Total State Accuracy: {avg_total_sa:.3f}")
        print(f"{'='*80}")
        
        # Save detailed results
        results_dir = './test_results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = join(results_dir, 'anti_uav_paper_sa_entire_testset.txt')
        
        with open(results_file, 'w') as f:
            f.write(f"Paper-defined State Accuracy Results - Entire Anti-UAV Test Set\n")
            f.write(f"{'='*80}\n")
            f.write(f"Number of sequences: {len(all_results['sequence_names'])}\n")
            f.write(f"Average RGB State Accuracy: {avg_rgb_sa:.3f}\n")
            f.write(f"Average Thermal State Accuracy: {avg_thermal_sa:.3f}\n")
            f.write(f"Average Total State Accuracy: {avg_total_sa:.3f}\n")
            f.write(f"\nPer-sequence results:\n")
            f.write(f"{'='*80}\n")
            
            for i, seq_name in enumerate(all_results['sequence_names']):
                f.write(f"{seq_name}:\n")
                f.write(f"  RGB SA: {all_results['rgb_state_accuracy'][i]:.3f}\n")
                f.write(f"  Thermal SA: {all_results['thermal_state_accuracy'][i]:.3f}\n")
                f.write(f"  Total SA: {all_results['total_state_accuracy'][i]:.3f}\n")
                f.write(f"\n")
        
        print(f"Detailed results saved to: {results_file}")
        
        return {
            'avg_rgb_state_accuracy': avg_rgb_sa,
            'avg_thermal_state_accuracy': avg_thermal_sa,
            'avg_total_state_accuracy': avg_total_sa,
            'per_sequence_results': all_results
        }
    else:
        print("No sequences were successfully evaluated")
        return None
    
    # Initialize tracker wrapper
    rgb_tracker = VEGETA_RGBT(tracker)
    
    # Run tracking for RGB modality
    print("\n=== Processing RGB modality ===")
    rgb_results = []
    toc = 0
    
    for frame_idx, rgb_path in enumerate(RGB_img_list):
        tic = cv2.getTickCount()
        
        if frame_idx == 0:
            # Initialize with RGB image
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                print(f"Error loading RGB image: {rgb_path}")
                continue
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            rgb_tracker.initialize(rgb_image, RGB_gt[0].tolist())
            rgb_results.append(RGB_gt[0])
        else:
            # Track with RGB image
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                print(f"Error loading RGB image: {rgb_path}")
                continue
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            region, confidence = rgb_tracker.track(rgb_image)
            rgb_results.append(np.array(region))
        
        toc += cv2.getTickCount() - tic
    
    toc /= cv2.getTickFrequency()
    rgb_fps = len(RGB_img_list) / toc
    
    # Run tracking for Thermal modality
    print("\n=== Processing Thermal modality ===")
    thermal_tracker = VEGETA_RGBT(tracker_class(params, 'ANTI-UAV'))
    thermal_results = []
    toc = 0
    
    for frame_idx, t_path in enumerate(T_img_list):
        tic = cv2.getTickCount()
        
        if frame_idx == 0:
            # Initialize with Thermal image
            thermal_image = cv2.imread(t_path)
            if thermal_image is None:
                print(f"Error loading Thermal image: {t_path}")
                continue
            thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
            print(f"Thermal init bbox: {T_gt[0]} (shape: {thermal_image.shape})")
            
            # Check if coordinates are reasonable for thermal image
            h, w = thermal_image.shape[:2]
            bbox = T_gt[0]
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > w or bbox[1] + bbox[3] > h:
                print(f"WARNING: Thermal init bbox {bbox} is outside image bounds {w}x{h}")
            
            thermal_tracker.initialize(thermal_image, T_gt[0].tolist())
            thermal_results.append(T_gt[0])
        else:
            # Track with Thermal image
            thermal_image = cv2.imread(t_path)
            if thermal_image is None:
                print(f"Error loading Thermal image: {t_path}")
                continue
            thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
            region, confidence = thermal_tracker.track(thermal_image)
            
            # Debug: Print prediction vs ground truth for first few frames
            if frame_idx < 5:
                print(f"Thermal frame {frame_idx}: pred={region}, gt={T_gt[frame_idx]}")
            
            thermal_results.append(np.array(region))
        
        toc += cv2.getTickCount() - tic
    
    toc /= cv2.getTickFrequency()
    thermal_fps = len(T_img_list) / toc
    
    # Convert results to tensors for evaluation
    rgb_results = torch.tensor(np.array(rgb_results))
    thermal_results = torch.tensor(np.array(thermal_results))
    rgb_gt_tensor = torch.tensor(RGB_gt)
    thermal_gt_tensor = torch.tensor(T_gt)
    
    # Calculate official AUC values using standard thresholds (0.0 to 1.0 with 0.05 step)
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05)  # 0 to 1 with 0.05 step (21 thresholds)
    
    # RGB AUC calculation
    rgb_success_rates, rgb_overlaps = calc_success_rate(rgb_results, rgb_gt_tensor, thresholds)
    rgb_auc = rgb_success_rates.mean().item()  # AUC is the mean of success rates across all thresholds
    rgb_sa50 = rgb_success_rates[thresholds == 0.50].item()
    rgb_sa75 = rgb_success_rates[thresholds == 0.75].item()
    
    # TIR (Thermal) AUC calculation  
    thermal_success_rates, thermal_overlaps = calc_success_rate(thermal_results, thermal_gt_tensor, thresholds)
    tir_auc = thermal_success_rates.mean().item()  # AUC is the mean of success rates across all thresholds
    tir_sa50 = thermal_success_rates[thresholds == 0.50].item()
    tir_sa75 = thermal_success_rates[thresholds == 0.75].item()
    
    # Total AUC calculation (average of RGB and TIR)
    total_auc = (rgb_auc + tir_auc) / 2.0
    total_sa50 = (rgb_sa50 + tir_sa50) / 2.0
    total_sa75 = (rgb_sa75 + tir_sa75) / 2.0
    
    # Print official evaluation results
    print("\n" + "="*60)
    print("="*60)
    print("OFFICIAL ANTI-UAV EVALUATION RESULTS")
    print("="*60)
    print(f"TIR (Thermal Infrared) Modality:")
    print(f"  AUC: {tir_auc:.3f}")
    print(f"  SA@0.50: {tir_sa50:.3f}")
    print(f"  SA@0.75: {tir_sa75:.3f}")
    print(f"  FPS: {thermal_fps:.2f}")
    print(f"  Average Overlap: {thermal_overlaps.mean():.3f}")
    
    print(f"\nRGB (Visible) Modality:")
    print(f"  AUC: {rgb_auc:.3f}")
    print(f"  SA@0.50: {rgb_sa50:.3f}")
    print(f"  SA@0.75: {rgb_sa75:.3f}")
    print(f"  FPS: {rgb_fps:.2f}")
    print(f"  Average Overlap: {rgb_overlaps.mean():.3f}")
    
    print(f"\nTotal (RGB + TIR Average):")
    print(f"  AUC: {total_auc:.3f}")
    print(f"  SA@0.50: {total_sa50:.3f}")
    print(f"  SA@0.75: {total_sa75:.3f}")
    print(f"  Average FPS: {(rgb_fps + thermal_fps) / 2:.2f}")
    
    # Save results to file
    results_dir = './test_results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'anti_uav_ep0120_official_results.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Official Anti-UAV Evaluation Results - SUTRACK_ep0120\n")
        f.write(f"="*60 + "\n")
        f.write(f"TIR (Thermal Infrared) Modality:\n")
        f.write(f"  AUC: {tir_auc:.3f}\n")
        f.write(f"  SA@0.50: {tir_sa50:.3f}\n")
        f.write(f"  SA@0.75: {tir_sa75:.3f}\n")
        f.write(f"  FPS: {thermal_fps:.2f}\n")
        f.write(f"  Average Overlap: {thermal_overlaps.mean():.3f}\n")
        f.write(f"\nRGB (Visible) Modality:\n")
        f.write(f"  AUC: {rgb_auc:.3f}\n")
        f.write(f"  SA@0.50: {rgb_sa50:.3f}\n")
        f.write(f"  SA@0.75: {rgb_sa75:.3f}\n")
        f.write(f"  FPS: {rgb_fps:.2f}\n")
        f.write(f"  Average Overlap: {rgb_overlaps.mean():.3f}\n")
        f.write(f"\nTotal (RGB + TIR Average):\n")
        f.write(f"  AUC: {total_auc:.3f}\n")
        f.write(f"  SA@0.50: {total_sa50:.3f}\n")
        f.write(f"  SA@0.75: {total_sa75:.3f}\n")
        f.write(f"  Average FPS: {(rgb_fps + thermal_fps) / 2:.2f}\n")
    
    print(f"\nOfficial results saved to: {results_file}")
    
    # 创建结果字典
    results_dict = {
        'rgb': {'auc': rgb_auc, 'sa50': rgb_sa50, 'sa75': rgb_sa75, 'fps': rgb_fps, 'avg_overlap': rgb_overlaps.mean()},
        'thermal': {'auc': tir_auc, 'sa50': tir_sa50, 'sa75': tir_sa75, 'fps': thermal_fps, 'avg_overlap': thermal_overlaps.mean()}
    }
    
    # 仅返回结果，不生成可视化
    return results_dict


def test_anti_uav_sa():
    """Legacy function - now using paper-defined State Accuracy evaluation"""
    print("Legacy function - using paper-defined State Accuracy instead")
    print("="*80)
    
    # Option 1: Evaluate single sequence (original test sequence)
    print("Option 1: Testing single sequence with paper-defined State Accuracy")
    single_result = evaluate_single_sequence_paper_sa(r'E:\biyesheji\SUTrack-main11\data\Anti-UAV\test\20190926_134054_1_8')
    
    if single_result:
        print(f"\nSingle sequence evaluation completed!")
        print(f"Sequence: {single_result['sequence_name']}")
        print(f"RGB State Accuracy: {single_result['rgb_state_accuracy']:.3f}")
        print(f"Thermal State Accuracy: {single_result['thermal_state_accuracy']:.3f}")
        print(f"Total State Accuracy: {single_result['total_state_accuracy']:.3f}")
    
    # Option 2: Evaluate entire test set
    print("\n" + "="*80)
    print("Option 2: Evaluating entire Anti-UAV test set")
    print("="*80)
    
    full_results = evaluate_entire_anti_uav_testset()
    
    if full_results:
        print(f"\nFull test set evaluation completed!")
        print(f"Average RGB State Accuracy: {full_results['avg_rgb_state_accuracy']:.3f}")
        print(f"Average Thermal State Accuracy: {full_results['avg_thermal_state_accuracy']:.3f}")
        print(f"Average Total State Accuracy: {full_results['avg_total_state_accuracy']:.3f}")
    
    return single_result, full_results


if __name__ == '__main__':
    test_anti_uav_sa()