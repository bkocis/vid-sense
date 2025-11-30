"""
Ground Truth Evaluator for Driver Monitoring System

This module provides functionality to:
- Load ground truth data for videos
- Evaluate detections against ground truth at frame level
- Calculate accuracy metrics (precision, recall, F1, etc.)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GroundTruthEvaluator:
    """
    Evaluates driver monitoring detections against ground truth data.
    """
    
    def __init__(self, ground_truth_file: Optional[str] = None):
        """
        Initialize the evaluator with ground truth data.
        
        Args:
            ground_truth_file: Path to ground truth JSON file. 
                              If None, looks for data/ground_truth.json
        """
        if ground_truth_file is None:
            # Default to data/ground_truth.json relative to project root
            project_root = Path(__file__).parent.parent
            ground_truth_file = project_root / "data" / "ground_truth.json"
        
        self.ground_truth_file = Path(ground_truth_file)
        self.ground_truth_data = self._load_ground_truth()
    
    def _load_ground_truth(self) -> Dict:
        """
        Load ground truth data from JSON file.
        
        Returns:
            Dictionary with ground truth data
        """
        if not self.ground_truth_file.exists():
            logger.warning(f"Ground truth file not found: {self.ground_truth_file}")
            return {}
        
        try:
            with open(self.ground_truth_file, 'r') as f:
                data = json.load(f)
                return data.get('ground_truth', {})
        except Exception as e:
            logger.error(f"Error loading ground truth file: {e}")
            return {}
    
    def get_ground_truth_for_video(self, video_filename: str) -> Optional[List[Dict]]:
        """
        Get ground truth impaired periods for a video.
        
        Args:
            video_filename: Name of the video file (e.g., "recording_20251130_134853_F.mp4")
        
        Returns:
            List of impaired periods with start_time and end_time, or None if not found
        """
        # Try exact match first
        if video_filename in self.ground_truth_data:
            return self.ground_truth_data[video_filename].get('impaired_periods', [])
        
        # Try matching just the filename without path
        video_name = Path(video_filename).name
        if video_name in self.ground_truth_data:
            return self.ground_truth_data[video_name].get('impaired_periods', [])
        
        return None
    
    def is_impaired_at_time(self, video_filename: str, timestamp: float) -> bool:
        """
        Check if a timestamp falls within any impaired period for the video.
        
        Args:
            video_filename: Name of the video file
            timestamp: Timestamp in seconds
        
        Returns:
            True if timestamp is within an impaired period, False otherwise
        """
        impaired_periods = self.get_ground_truth_for_video(video_filename)
        if impaired_periods is None:
            return False
        
        for period in impaired_periods:
            start = period['start_time']
            end = period['end_time']
            if start <= timestamp <= end:
                return True
        
        return False
    
    def calculate_frame_level_accuracy(
        self,
        video_filename: str,
        detections: List[Dict],
        fps: float
    ) -> Dict[str, any]:
        """
        Calculate frame-level accuracy metrics.
        
        Args:
            video_filename: Name of the video file
            detections: List of detection dictionaries, each containing:
                       - 'video_frame_number': frame number
                       - 'alert_level': 'LOW', 'MODERATE', or 'HIGH'
                       - 'timestamp' (optional): timestamp in seconds
            fps: Frames per second of the video
        
        Returns:
            Dictionary with accuracy metrics
        """
        impaired_periods = self.get_ground_truth_for_video(video_filename)
        
        if impaired_periods is None:
            logger.warning(f"No ground truth found for {video_filename}")
            return {
                'has_ground_truth': False,
                'message': f'No ground truth data available for {video_filename}'
            }
        
        # Convert impaired periods to frame ranges
        impaired_frames = set()
        for period in impaired_periods:
            start_frame = int(period['start_time'] * fps)
            end_frame = int(period['end_time'] * fps)
            # Include all frames in the range (inclusive)
            for frame_num in range(start_frame, end_frame + 1):
                impaired_frames.add(frame_num)
        
        # Process detections
        true_positives = 0  # Detected impaired when actually impaired
        false_positives = 0  # Detected impaired when not impaired
        false_negatives = 0  # Did not detect impaired when actually impaired
        true_negatives = 0  # Did not detect impaired when not impaired
        
        # Track which frames we've evaluated
        evaluated_frames = set()
        
        # Evaluate each detection
        for detection in detections:
            frame_num = detection.get('video_frame_number')
            if frame_num is None:
                continue
            
            # Calculate timestamp if not provided
            timestamp = detection.get('timestamp')
            if timestamp is None:
                timestamp = frame_num / fps if fps > 0 else 0
            
            # Determine if this frame is actually impaired (ground truth)
            is_actually_impaired = frame_num in impaired_frames
            
            # Determine if system detected impairment
            # Consider MODERATE and HIGH as impaired detections
            alert_level = detection.get('alert_level', 'LOW')
            is_detected_impaired = alert_level in ['MODERATE', 'HIGH']
            
            # Also check if the detection is from impaired_driving analysis
            if 'analyses' in detection:
                impaired_analysis = detection['analyses'].get('impaired_driving', {})
                impaired_alert_level = impaired_analysis.get('alert_level', 'LOW')
                is_detected_impaired = impaired_alert_level in ['MODERATE', 'HIGH']
            
            # Calculate metrics
            if is_actually_impaired and is_detected_impaired:
                true_positives += 1
            elif not is_actually_impaired and is_detected_impaired:
                false_positives += 1
            elif is_actually_impaired and not is_detected_impaired:
                false_negatives += 1
            else:
                true_negatives += 1
            
            evaluated_frames.add(frame_num)
        
        # Calculate false negatives for frames we didn't evaluate but should have
        # (frames in impaired periods that weren't in our detections)
        for frame_num in impaired_frames:
            if frame_num not in evaluated_frames:
                false_negatives += 1
        
        # Calculate metrics
        total_impaired_frames = len(impaired_frames)
        total_evaluated = len(evaluated_frames)
        
        # Calculate total impaired time (sum of all impaired period durations)
        total_impaired_time = sum(period['end_time'] - period['start_time'] for period in impaired_periods)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives) if (true_positives + false_positives + true_negatives + false_negatives) > 0 else 0.0
        
        logger.info(f"Total impaired time: {total_impaired_time:.2f}s ({total_impaired_time/60:.2f} minutes)")
        
        return {
            'has_ground_truth': True,
            'video_filename': video_filename,
            'total_impaired_frames': total_impaired_frames,
            'total_impaired_time': total_impaired_time,
            'total_evaluated_frames': total_evaluated,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'impaired_periods': impaired_periods
        }
    
    def print_accuracy_report(self, metrics: Dict[str, any]) -> None:
        """
        Print a formatted accuracy report.
        
        Args:
            metrics: Dictionary with accuracy metrics from calculate_frame_level_accuracy
        """
        if not metrics.get('has_ground_truth', False):
            print(f"\n⚠️  {metrics.get('message', 'No ground truth available')}")
            return
        
        print("\n" + "="*80)
        print("GROUND TRUTH ACCURACY REPORT")
        print("="*80)
        print(f"Video: {metrics['video_filename']}")
        print(f"\nGround Truth Impaired Periods:")
        for i, period in enumerate(metrics['impaired_periods'], 1):
            duration = period['end_time'] - period['start_time']
            print(f"  {i}. {period['start_time']:.2f}s - {period['end_time']:.2f}s (duration: {duration:.2f}s)")
        
        print(f"\nFrame-Level Metrics:")
        print(f"  Total impaired frames (ground truth): {metrics['total_impaired_frames']}")
        print(f"  Total impaired time: {metrics.get('total_impaired_time', 0):.2f}s ({metrics.get('total_impaired_time', 0)/60:.2f} minutes)")
        print(f"  Total evaluated frames: {metrics['total_evaluated_frames']}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP):  {metrics['true_positives']}")
        print(f"  False Positives (FP): {metrics['false_positives']}")
        print(f"  False Negatives (FN): {metrics['false_negatives']}")
        print(f"  True Negatives (TN):  {metrics['true_negatives']}")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print("="*80 + "\n")
    
    def save_evaluation_results(self, metrics: Dict[str, any], video_path: Optional[str] = None) -> Optional[Path]:
        """
        Save evaluation results to a JSON file next to ground_truth.json.
        
        Args:
            metrics: Dictionary with accuracy metrics from calculate_frame_level_accuracy
            video_path: Optional path to the video file (used to determine output filename)
        
        Returns:
            Path to the saved file, or None if not saved
        """
        if not metrics.get('has_ground_truth', False):
            logger.debug("No ground truth available, skipping save")
            return None
        
        # Determine output file path
        if video_path:
            video_filename = Path(video_path).name
            video_stem = Path(video_path).stem
        else:
            video_filename = metrics.get('video_filename', 'unknown')
            video_stem = Path(video_filename).stem
        
        # Save next to ground_truth.json
        output_dir = self.ground_truth_file.parent
        output_file = output_dir / f"{video_stem}_evaluation.json"
        
        # Prepare results for saving
        import datetime
        results = {
            'video_filename': metrics['video_filename'],
            'evaluation_timestamp': datetime.datetime.now().isoformat(),
            'ground_truth_periods': metrics['impaired_periods'],
            'total_impaired_time_seconds': metrics.get('total_impaired_time', 0),
            'total_impaired_time_minutes': metrics.get('total_impaired_time', 0) / 60,
            'metrics': {
                'total_impaired_frames': metrics['total_impaired_frames'],
                'total_impaired_time': metrics.get('total_impaired_time', 0),
                'total_evaluated_frames': metrics['total_evaluated_frames'],
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'true_negatives': metrics['true_negatives'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy']
            },
            'metrics_percentage': {
                'precision': metrics['precision'] * 100,
                'recall': metrics['recall'] * 100,
                'f1_score': metrics['f1_score'] * 100,
                'accuracy': metrics['accuracy'] * 100
            }
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            return None

