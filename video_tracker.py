"""
SUTrack Video Object Tracking Program
Supports video file input, real-time tracking with visualization, and output saving.
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path

# Add project path
prj = os.path.dirname(os.path.abspath(__file__))
if prj not in sys.path:
    sys.path.insert(0, prj)

from lib.test.parameter.sutrack import parameters
from lib.test.tracker.sutrack import SUTRACK
from lib.test.tracker.utils import Preprocessor
from lib.test.tracker.utils import sample_target


class VideoTracker:
    """
    Video Object Tracker using SUTrack model
    Supports video file input, visualization, and result saving
    """
    
    def __init__(self, model_path, yaml_name='sutrack_b224', dataset_name='GOT10K', 
                 enhance_infrared=False, search_factor=None, update_interval=None):
        """
        Initialize the tracker with specified model
        
        Args:
            model_path: Path to the model checkpoint file
            yaml_name: Configuration yaml file name (default: sutrack_b224)
            dataset_name: Dataset name for tracker configuration
            enhance_infrared: Whether to enhance infrared image contrast
            search_factor: Search area factor (default: 4.0, increase for heavy background)
            update_interval: Template update interval (default: 25, decrease for dynamic scenes)
        """
        self.model_path = model_path
        self.yaml_name = yaml_name
        self.dataset_name = dataset_name
        self.enhance_infrared = enhance_infrared
        
        # Load parameters
        self.params = parameters(yaml_name)
        self.params.checkpoint = model_path
        self.params.debug = False
        
        # Adjust parameters for heavy background interference
        if search_factor is not None:
            self.params.search_factor = search_factor
        if update_interval is not None:
            self.params.update_intervals = {'DEFAULT': update_interval}
        
        # Create tracker
        self.tracker = SUTRACK(self.params, dataset_name)
        self.initialized = False
        
        print(f"Model loaded successfully: {model_path}")
        print(f"Using config: {yaml_name}")
        print(f"Template size: {self.params.template_size}")
        print(f"Search size: {self.params.search_size}")
        print(f"Search factor: {self.params.search_factor}")
        if enhance_infrared:
            print("Infrared enhancement: ENABLED")
        
    def initialize(self, frame, bbox):
        """
        Initialize tracker with first frame and target bounding box
        
        Args:
            frame: First frame of video (numpy array)
            bbox: Initial bounding box [x, y, w, h]
        """
        # Convert BGR to RGB or handle grayscale/infrared
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif len(frame.shape) == 2:
            # Grayscale/infrared image - keep as is (will be converted to 3-channel in preprocessor)
            frame_rgb = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Single channel image with explicit dimension
            frame_rgb = frame.squeeze()
        else:
            frame_rgb = frame
            
        init_info = {'init_bbox': list(bbox)}
        self.tracker.initialize(frame_rgb, init_info)
        self.initialized = True
        print(f"Tracker initialized with bbox: {bbox}")
        
    def track(self, frame):
        """
        Track target in the given frame
        
        Args:
            frame: Current frame (numpy array)
            
        Returns:
            dict: Contains 'target_bbox' [x, y, w, h] and 'best_score' confidence
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
            
        # Convert BGR to RGB or handle grayscale/infrared
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif len(frame.shape) == 2:
            # Grayscale/infrared image - keep as is (will be converted to 3-channel in preprocessor)
            frame_rgb = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Single channel image with explicit dimension
            frame_rgb = frame.squeeze()
        else:
            frame_rgb = frame
            
        outputs = self.tracker.track(frame_rgb)
        return outputs
    
    def draw_bbox(self, frame, bbox, score=None, color=(0, 255, 0), thickness=2, 
                  score_threshold=0.3, lost_color=(0, 0, 255)):
        """
        Draw bounding box on frame
        
        Args:
            frame: Input frame
            bbox: Bounding box [x, y, w, h]
            score: Confidence score (optional)
            color: Box color (BGR) when target is tracked
            thickness: Line thickness
            score_threshold: Threshold for determining target loss (default: 0.3)
            lost_color: Box color when target is lost (default: red)
            
        Returns:
            Frame with drawn bbox (or original frame if target lost)
        """
        # Convert score to float
        score_val = None
        if score is not None:
            if hasattr(score, 'item'):
                score_val = score.item()
            else:
                score_val = float(score)
        
        # Check if target is lost (score below threshold)
        is_lost = score_val is not None and score_val < score_threshold
        
        # If target is lost, only display "LOST" text without bounding box
        if is_lost:
            # Display "TARGET LOST" warning at top-left corner
            lost_text = f"TARGET LOST (Score: {score_val:.3f})"
            cv2.putText(frame, lost_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, lost_color, 2)
            return frame
        
        # Normal case: draw bounding box
        x, y, w, h = [int(v) for v in bbox]
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label with score
        if score_val is not None:
            label = f"Target: {score_val:.3f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y - 10, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(frame, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0], label_y + 5), color, -1)
            # Draw label text
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Target", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
                       
        return frame
    
    def process_video(self, input_path, output_path=None, init_bbox=None, 
                     display=True, save_video=True, scale_factor=1.0):
        """
        Process entire video with tracking
        
        Args:
            input_path: Path to input video file
            output_path: Path to save output video (optional)
            init_bbox: Initial bounding box [x, y, w, h] (if None, will use interactive selection)
            display: Whether to display video in real-time
            save_video: Whether to save output video
            scale_factor: Scale factor for processing (1.0 = original size)
            
        Returns:
            list: List of tracking results for each frame
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Apply scale factor
        if scale_factor != 1.0:
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            print(f"Scaled to: {width}x{height}")
        
        # Setup video writer
        writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output video will be saved to: {output_path}")
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame from video")
            
        # Resize frame if needed
        if scale_factor != 1.0:
            frame = cv2.resize(frame, (width, height))
        
        # Get initial bbox if not provided
        if init_bbox is None:
            init_bbox = self.select_roi_interactive(frame)
            if init_bbox is None:
                print("No ROI selected. Exiting.")
                cap.release()
                return []
        
        # Initialize tracker
        self.initialize(frame, init_bbox)
        
        # Draw initial bbox
        frame = self.draw_bbox(frame.copy(), init_bbox, color=(0, 255, 255), thickness=2)
        
        # Process frames
        results = []
        frame_count = 0
        
        print("Starting tracking...")
        
        while True:
            # Process current frame
            if frame_count > 0:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame if needed
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, (width, height))
                
                # Track
                outputs = self.track(frame)
                bbox = outputs['target_bbox']
                score = outputs['best_score']
                
                # Convert score to float for storage and display
                score_val = score.item() if hasattr(score, 'item') else float(score)
                
                results.append({
                    'frame': frame_count,
                    'bbox': bbox,
                    'score': score_val
                })
                
                # Draw results (with target loss detection)
                frame = self.draw_bbox(frame.copy(), bbox, score_val, color=(0, 255, 0), 
                                      score_threshold=0.47, lost_color=(0, 0, 255))
                
                # Add frame info (only show score if target not lost)
                if score_val >= 0.3:
                    info_text = f"Frame: {frame_count}/{total_frames} | Score: {score_val:.3f}"
                else:
                    info_text = f"Frame: {frame_count}/{total_frames} | TARGET LOST"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            else:
                # First frame - add info
                info_text = f"Frame: {frame_count}/{total_frames} | Initialized"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
                results.append({
                    'frame': frame_count,
                    'bbox': init_bbox,
                    'score': 1.0
                })
            
            # Write to output video
            if writer:
                writer.write(frame)
            
            # Display
            if display:
                cv2.imshow('SUTrack Video Tracking', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Tracking stopped by user")
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)  # Pause on space
            
            # Progress
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"Tracking completed. Processed {frame_count} frames.")
        return results
    
    def select_roi_interactive(self, frame):
        """
        Interactive ROI selection using OpenCV
        
        Args:
            frame: First frame of video
            
        Returns:
            Selected bounding box [x, y, w, h] or None
        """
        print("Please select the target region to track...")
        print("Draw a rectangle around the target and press SPACE or ENTER to confirm")
        print("Press 'c' to cancel")
        
        # Enhance infrared/grayscale image for better visibility
        display_frame = self._enhance_for_display(frame)
        
        # Create window
        cv2.namedWindow('Select Target', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Target', display_frame.shape[1], display_frame.shape[0])
        
        # Select ROI
        bbox = cv2.selectROI('Select Target', display_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('Select Target')
        
        if bbox[2] == 0 or bbox[3] == 0:
            return None
            
        return [bbox[0], bbox[1], bbox[2], bbox[3]]
    
    def _enhance_for_display(self, frame):
        """
        Enhance infrared/grayscale image for better display and selection
        
        Args:
            frame: Input frame (could be grayscale or BGR)
            
        Returns:
            Enhanced frame for display
        """
        # Check if grayscale/infrared
        if len(frame.shape) == 2:
            # Apply histogram equalization for better contrast
            enhanced = cv2.equalizeHist(frame.astype(np.uint8))
            # Convert to BGR for display
            display_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Single channel with explicit dimension
            enhanced = cv2.equalizeHist(frame.squeeze().astype(np.uint8))
            display_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            # Color image, no enhancement needed
            display_frame = frame
            
        return display_frame
    
    def _suppress_background(self, frame, bbox):
        """
        Suppress background interference by applying adaptive thresholding
        
        Args:
            frame: Input frame (grayscale or color)
            bbox: Target bounding box [x, y, w, h]
            
        Returns:
            Frame with suppressed background
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Get target region
        x, y, w, h = [int(v) for v in bbox]
        x = max(0, x)
        y = max(0, y)
        x2 = min(gray.shape[1], x + w)
        y2 = min(gray.shape[0], y + h)
        
        if x2 > x and y2 > y:
            target_region = gray[y:y2, x:x2]
            target_mean = np.mean(target_region)
            target_std = np.std(target_region)
            
            # Create mask based on target statistics
            lower_bound = max(0, target_mean - 2 * target_std)
            upper_bound = min(255, target_mean + 2 * target_std)
            
            mask = cv2.inRange(gray, int(lower_bound), int(upper_bound))
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Apply mask to original frame
            if len(frame.shape) == 3:
                result = cv2.bitwise_and(frame, frame, mask=mask)
            else:
                result = cv2.bitwise_and(gray, gray, mask=mask)
            
            return result
        
        return frame
    
    def process_webcam(self, camera_id=0, output_path=None, init_bbox=None):
        """
        Process webcam input for real-time tracking
        
        Args:
            camera_id: Camera device ID (default: 0)
            output_path: Path to save output video (optional)
            init_bbox: Initial bounding box (if None, will use interactive selection)
            
        Returns:
            list: List of tracking results
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
            
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        
        print(f"Camera resolution: {width}x{height}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Cannot read from camera")
        
        # Get initial bbox
        if init_bbox is None:
            init_bbox = self.select_roi_interactive(frame)
            if init_bbox is None:
                print("No ROI selected. Exiting.")
                cap.release()
                return []
        
        # Initialize tracker
        self.initialize(frame, init_bbox)
        
        # Process frames
        results = []
        frame_count = 0
        
        print("Starting real-time tracking... Press 'q' to quit, 'r' to reinitialize")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track
            outputs = self.track(frame)
            bbox = outputs['target_bbox']
            score = outputs['best_score']
            
            # Convert score to float for storage and display
            score_val = score.item() if hasattr(score, 'item') else float(score)
            
            results.append({
                'frame': frame_count,
                'bbox': bbox,
                'score': score_val
            })
            
            # Draw results (with target loss detection)
            display_frame = frame.copy()
            display_frame = self.draw_bbox(display_frame, bbox, score_val, color=(0, 255, 0),
                                          score_threshold=0.3, lost_color=(0, 0, 255))
            
            # Add info
            if score_val >= 0.3:
                info_text = f"Frame: {frame_count} | Score: {score_val:.3f}"
            else:
                info_text = f"Frame: {frame_count} | TARGET LOST"
            cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            
            # Write to output
            if writer:
                writer.write(display_frame)
            
            # Display
            cv2.imshow('SUTrack Real-time Tracking', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reinitialize
                print("Reinitializing tracker...")
                bbox = self.select_roi_interactive(frame)
                if bbox is not None:
                    self.initialize(frame, bbox)
                    print("Tracker reinitialized")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"Tracking completed. Processed {frame_count} frames.")
        return results


def main():
    parser = argparse.ArgumentParser(description='SUTrack Video Object Tracking')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file path or "webcam" for camera input')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--model', '-m', type=str, 
                       default=r'E:\biyesheji\SUTrack-main11\output\b224_antiuav_clahe\checkpoints\train\sutrack\sutrack_b224_antiuav_clahe\SUTRACK_ep0090.pth.tar',
                       help='Path to model checkpoint')
    parser.add_argument('--config', '-c', type=str, default='sutrack_b224_antiuav_clahe',
                       help='Configuration yaml name (sutrack_b224, sutrack_t224, etc.)')
    parser.add_argument('--bbox', '-b', type=str, default=None,
                       help='Initial bounding box as x,y,w,h (e.g., "100,100,50,50")')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                       help='Scale factor for processing (0.5 for half resolution)')
    parser.add_argument('--enhance-infrared', action='store_true',
                       help='Enable infrared image contrast enhancement')
    parser.add_argument('--search-factor', type=float, default=None,
                       help='Search area factor (default: 4.0, increase for heavy background)')
    parser.add_argument('--update-interval', type=int, default=None,
                       help='Template update interval (default: 25, decrease for dynamic scenes)')
    
    args = parser.parse_args()
    
    # Parse bbox if provided
    init_bbox = None
    if args.bbox:
        init_bbox = [float(x) for x in args.bbox.split(',')]
        print(f"Using provided bbox: {init_bbox}")
    
    # Determine output path
    output_path = None if args.no_save else args.output
    if output_path is None and not args.no_save and args.input != 'webcam':
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_tracked.mp4"
    
    # Create tracker with optional parameters for heavy background
    tracker = VideoTracker(
        model_path=args.model,
        yaml_name=args.config,
        dataset_name='GOT10K',
        enhance_infrared=args.enhance_infrared,
        search_factor=args.search_factor,
        update_interval=args.update_interval
    )
    
    # Process video or webcam
    try:
        if args.input.lower() == 'webcam':
            results = tracker.process_webcam(
                camera_id=0,
                output_path=output_path,
                init_bbox=init_bbox
            )
        else:
            results = tracker.process_video(
                input_path=args.input,
                output_path=output_path,
                init_bbox=init_bbox,
                display=not args.no_display,
                save_video=not args.no_save,
                scale_factor=args.scale
            )
        
        # Print summary
        if results:
            avg_score = sum(r['score'] for r in results) / len(results)
            print(f"\nTracking Summary:")
            print(f"  Total frames: {len(results)}")
            print(f"  Average confidence: {avg_score:.3f}")
            
            # Save results to text file
            if output_path:
                result_file = output_path.replace('.mp4', '_results.txt')
                with open(result_file, 'w') as f:
                    f.write("frame,x,y,w,h,score\n")
                    for r in results:
                        bbox = r['bbox']
                        f.write(f"{r['frame']},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{r['score']}\n")
                print(f"  Results saved to: {result_file}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()