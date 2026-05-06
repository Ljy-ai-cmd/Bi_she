"""
SUTrack MOT Model Video Tracking with Visualization
基于改进的MOT模型 (TTE + STCA + TMB) 进行红外无人机跟踪可视化
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse

prj = os.path.dirname(os.path.abspath(__file__))
if prj not in sys.path:
    sys.path.insert(0, prj)

from lib.test.parameter.sutrack import parameters
from lib.test.tracker.sutrack import SUTRACK


class MOTVideoTracker:
    def __init__(self, model_path, yaml_name='sutrack_b224_antiuav_mot', dataset_name='GOT10K',
                 search_factor=None, update_interval=None, score_threshold=0.3):
        self.model_path = model_path
        self.yaml_name = yaml_name
        self.dataset_name = dataset_name
        self.score_threshold = score_threshold

        self.params = parameters(yaml_name)
        self.params.checkpoint = model_path
        self.params.debug = False

        if search_factor is not None:
            self.params.search_factor = search_factor
        if update_interval is not None:
            self.params.update_intervals = {'DEFAULT': update_interval}

        self.tracker = SUTRACK(self.params, dataset_name)
        self.initialized = False

        print(f"MOT Model loaded: {model_path}")
        print(f"Config: {yaml_name}")
        print(f"Template size: {self.params.template_size}")
        print(f"Search size: {self.params.search_size}")
        print(f"Search factor: {self.params.search_factor}")
        print(f"Score threshold: {score_threshold}")

    def initialize(self, frame, bbox):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif len(frame.shape) == 2:
            frame_rgb = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame_rgb = frame.squeeze()
        else:
            frame_rgb = frame

        init_info = {'init_bbox': list(bbox)}
        self.tracker.initialize(frame_rgb, init_info)
        self.initialized = True
        print(f"MOT Tracker initialized with bbox: {bbox}")

    def track(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized.")

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif len(frame.shape) == 2:
            frame_rgb = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame_rgb = frame.squeeze()
        else:
            frame_rgb = frame

        return self.tracker.track(frame_rgb)

    def draw_bbox(self, frame, bbox, score=None, color=(0, 255, 0), thickness=2,
                  score_threshold=0.3, lost_color=(0, 0, 255)):
        score_val = None
        if score is not None:
            if hasattr(score, 'item'):
                score_val = score.item()
            else:
                score_val = float(score)

        is_lost = score_val is not None and score_val < score_threshold

        if is_lost:
            lost_text = f"TARGET LOST (Score: {score_val:.3f})"
            cv2.putText(frame, lost_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, lost_color, 2)
            return frame

        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        if score_val is not None:
            label = f"Target: {score_val:.3f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y - 10, label_size[1] + 10)
            cv2.rectangle(frame, (x, label_y - label_size[1] - 5),
                         (x + label_size[0], label_y + 5), color, -1)
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Target", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

        return frame

    def select_roi_interactive(self, frame):
        print("Draw a rectangle around the drone target and press SPACE/ENTER to confirm.")
        print("Press 'c' to cancel.")

        display_frame = self._enhance_for_display(frame)

        cv2.namedWindow('Select Drone Target', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Drone Target', display_frame.shape[1], display_frame.shape[0])

        bbox = cv2.selectROI('Select Drone Target', display_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('Select Drone Target')

        if bbox[2] == 0 or bbox[3] == 0:
            return None

        return [bbox[0], bbox[1], bbox[2], bbox[3]]

    def _enhance_for_display(self, frame):
        if len(frame.shape) == 2:
            enhanced = cv2.equalizeHist(frame.astype(np.uint8))
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            enhanced = cv2.equalizeHist(frame.squeeze().astype(np.uint8))
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return frame

    def process_video(self, input_path, output_path=None, init_bbox=None,
                     display=True, save_video=True, scale_factor=1.0):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        if scale_factor != 1.0:
            width = int(width * scale_factor)
            height = int(height * scale_factor)

        writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output: {output_path}")

        ret, frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame.")

        if scale_factor != 1.0:
            frame = cv2.resize(frame, (width, height))

        if init_bbox is None:
            init_bbox = self.select_roi_interactive(frame)
            if init_bbox is None:
                print("No ROI selected.")
                cap.release()
                return []

        self.initialize(frame, init_bbox)

        results = []
        frame_count = 0
        fps_list = []
        font_color = (0, 255, 255)
        bg_color = (0, 0, 0)

        print("Tracking started... Press 'q' to quit, SPACE to pause.")
        tracker_name = "MOT (TTE+STCA+TMB)"

        while True:
            if frame_count > 0:
                ret, frame = cap.read()
                if not ret:
                    break

                if scale_factor != 1.0:
                    frame = cv2.resize(frame, (width, height))

                t_start = cv2.getTickCount()
                outputs = self.track(frame)
                t_end = cv2.getTickCount()
                frame_time = (t_end - t_start) / cv2.getTickFrequency()
                fps_list.append(frame_time)

                bbox = outputs['target_bbox']
                score = outputs['best_score']
                score_val = score.item() if hasattr(score, 'item') else float(score)

                results.append({'frame': frame_count, 'bbox': bbox, 'score': score_val})

                frame = self.draw_bbox(frame.copy(), bbox, score_val, color=(0, 255, 0),
                                      score_threshold=self.score_threshold, lost_color=(0, 0, 255))

                avg_fps = 1.0 / np.mean(fps_list[-30:]) if len(fps_list) > 0 else 0
                info_lines = [
                    f"{tracker_name}",
                    f"Frame: {frame_count}/{total_frames}",
                    f"Score: {score_val:.3f}",
                    f"FPS: {avg_fps:.1f}",
                ]
            else:
                info_lines = [
                    f"{tracker_name}",
                    f"Frame: {frame_count}/{total_frames}",
                    f"Initialized",
                ]
                results.append({'frame': frame_count, 'bbox': init_bbox, 'score': 1.0})

            display_frame = frame.copy()
            for i, line in enumerate(info_lines):
                y_pos = 25 + i * 24
                cv2.rectangle(display_frame, (5, y_pos - 18), (350, y_pos + 6), bg_color, -1)
                cv2.putText(display_frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                           0.55, font_color, 2)

            if writer:
                writer.write(display_frame)

            if display:
                cv2.imshow('MOT SUTrack - Infrared Drone Tracking', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Tracking stopped by user.")
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)

            frame_count += 1
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        avg_fps = 1.0 / np.mean(fps_list) if fps_list else 0
        print(f"\nTracking completed. {frame_count} frames, avg FPS: {avg_fps:.1f}")

        if results and output_path:
            result_file = output_path.replace('.mp4', '_results.txt')
            with open(result_file, 'w') as f:
                f.write("frame,x,y,w,h,score\n")
                for r in results:
                    b = r['bbox']
                    f.write(f"{r['frame']},{b[0]},{b[1]},{b[2]},{b[3]},{r['score']}\n")
            print(f"Results saved to: {result_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description='MOT SUTrack - Infrared Drone Tracking Visualization')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--model', '-m', type=str,
                       default=r'E:\biyesheji\SUTrack-main11\output\b224_antiuav_mot_v2\checkpoints\train\sutrack\sutrack_b224_antiuav_mot_v2\SUTRACK_ep0160.pth.tar',
                       help='Path to model checkpoint')
    parser.add_argument('--config', '-c', type=str, default='sutrack_b224_antiuav_mot_v2',
                       help='Configuration yaml name')
    parser.add_argument('--bbox', '-b', type=str, default=None,
                       help='Initial bbox: x,y,w,h (e.g. "100,100,50,50")')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                       help='Scale factor (0.5 = half resolution)')
    parser.add_argument('--search-factor', type=float, default=None,
                       help='Search area factor (default: 4.0)')
    parser.add_argument('--update-interval', type=int, default=None,
                       help='Template update interval (default: 25)')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                       help='Score threshold for target loss detection')

    args = parser.parse_args()

    init_bbox = None
    if args.bbox:
        init_bbox = [float(x) for x in args.bbox.split(',')]
        print(f"Using provided bbox: {init_bbox}")

    output_path = None if args.no_save else args.output
    if output_path is None and not args.no_save:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_mot_tracked.mp4"

    tracker = MOTVideoTracker(
        model_path=args.model,
        yaml_name=args.config,
        dataset_name='GOT10K',
        search_factor=args.search_factor,
        update_interval=args.update_interval,
        score_threshold=args.score_threshold,
    )

    try:
        results = tracker.process_video(
            input_path=args.input,
            output_path=output_path,
            init_bbox=init_bbox,
            display=not args.no_display,
            save_video=not args.no_save,
            scale_factor=args.scale,
        )

        if results:
            avg_score = sum(r['score'] for r in results) / len(results)
            print(f"\nSummary:")
            print(f"  Frames: {len(results)}")
            print(f"  Avg score: {avg_score:.3f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
