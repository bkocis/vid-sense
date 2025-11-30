"""
Fast In-Car Driver Monitoring System
Optimized for speed with single comprehensive analysis

Key Features:
- Single LLM query per frame sequence (fastest response)
- Reduced frame buffer for lower latency
- Concise analysis focused on safety-critical information
- Minimal processing overhead
- Fast alert detection

Optimizations:
- Single query instead of 5 separate analyses (5x faster)
- Smaller frame buffer (3 frames vs 5)
- Shorter response length (128 tokens vs 256)
- Reduced context window (1024 vs 2048)
"""

import cv2
import ollama
import datetime
import logging
from typing import List, Optional, Dict
from collections import deque
import time
import os
import argparse
from pathlib import Path
from ground_truth_evaluator import GroundTruthEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastDriverMonitoringConfig:
    """Configuration for fast driver monitoring system"""
    MODEL_NAME = "llava:7b-v1.6-mistral-q2_K"
    FRAME_RATE = 30
    PROCESSING_INTERVAL_SECONDS = 1  # Process every 1 second (faster than standard)
    FRAME_BUFFER_SIZE = 3  # Keep last 3 frames (reduced for speed)
    
    # LLM generation parameters (optimized for speed)
    TEMPERATURE = 0
    TOP_K = 1
    TOP_P = 0.1
    NUM_CTX = 1024  # Reduced context for faster processing
    NUM_PREDICT = 256  # Increased to ensure full analysis responses


class FastDriverAnalyzer:
    """
    Fast analyzer with single comprehensive query.
    Returns quick description of driver state and safety concerns.
    """
    
    def __init__(self, config: FastDriverMonitoringConfig):
        self.config = config
        self.alert_history = []
    
    def analyze_driver_state(self, frames: List[bytes]) -> Dict[str, any]:
        """
        Single comprehensive analysis of driver state.
        Combines all analyses into one fast query.
        
        Args:
            frames: List of frame bytes for temporal analysis
        
        Returns:
            Dictionary with analysis results
        """
        query = (
            "Analyze these sequential video frames of a driver in a car for safety-critical indicators. "
            "Focus specifically on CLOSED EYES and FATIGUE detection. "
            ""
            "You MUST provide a complete analysis with the following format: "
            ""
            "First, state the concern level in ALL CAPS on its own line: LOW, MODERATE, or HIGH"
            ""
            "Then provide a detailed description (2-3 sentences) covering: "
            "1) EYE STATE: Are the driver's eyes fully open, partially closed, or completely closed? "
            "   Look for: eyes closed for more than 1 second, heavy eyelids, drooping eyes, slow blinking, or eyes struggling to stay open. "
            "2) FATIGUE INDICATORS: Check for signs of drowsiness or fatigue such as: "
            "   - Head nodding or falling forward "
            "   - Yawning "
            "   - Rubbing eyes "
            "   - Slouched posture or head tilting "
            "   - Slow or delayed reactions "
            "3) Overall alertness: Is the driver awake, alert, and attentive, or showing signs of sleepiness/tiredness? "
            "4) Hand position: Briefly note if hands are on steering wheel (secondary concern). "
            ""
            "CRITICAL RULES: "
            "- If eyes are CLOSED or HEAVILY DROOPING, this is HIGH concern. "
            "- If there are clear fatigue indicators (yawning, head nodding, eyes struggling to stay open), this is MODERATE to HIGH concern. "
            "- If the driver appears normal, alert, and eyes are fully open, state 'LOW concern'. "
            ""
            "IMPORTANT: You must provide BOTH the concern level AND the detailed description. Do not skip the description!"
        )
        
        start_time = time.time()
        response = self._query_llm(query, frames)
        query_time = time.time() - start_time
        
        if response is None:
            response = "Error: Could not analyze driver state. LLM service unavailable."
            logger.error("Failed to analyze driver state: LLM query returned None")
            alert_level = 'LOW'
        else:
            # Log query time
            logger.info(f"LLM query completed in {query_time:.2f}s")
            alert_level = self._parse_alert_level(response)
            logger.info(f"Final alert level: {alert_level}")
            logger.info("="*80)
        
        result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'analysis_type': 'comprehensive',
            'response': response,
            'alert_level': alert_level,
            'frames_analyzed': len(frames),
            'query_time_seconds': query_time
        }
        
        # Trigger alert for concerning states
        if alert_level in ['MODERATE', 'HIGH']:
            self._trigger_alert(result)
        
        return result
    
    def _query_llm(self, query: str, image_list: List[bytes]) -> Optional[str]:
        """
        Query the LLaVA model with images and text query.
        
        Args:
            query: Text query/question about the images
            image_list: List of image bytes (JPEG encoded frames)
        
        Returns:
            Response content from the LLM, or None if error
        """
        try:
            res = ollama.chat(
                model=self.config.MODEL_NAME,
                options={
                    'temperature': self.config.TEMPERATURE,
                    'top_k': self.config.TOP_K,
                    'top_p': self.config.TOP_P,
                    'num_ctx': self.config.NUM_CTX,
                    'num_predict': self.config.NUM_PREDICT,
                },
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            "You are a fast driver monitoring system specialized in detecting closed eyes and fatigue. "
                            "Your primary focus is identifying when a driver's eyes are closed or showing signs of drowsiness. "
                            "You MUST always provide a complete response that includes: "
                            "1) The concern level (LOW, MODERATE, or HIGH) on its own line, "
                            "2) A detailed 2-3 sentence analysis describing the driver's eye state, fatigue indicators, alertness, and hand position. "
                            "Be objective: normal states with eyes fully open and alert are LOW concern. "
                            "Report MODERATE concern for fatigue indicators (yawning, head nodding, heavy eyelids). "
                            "Report HIGH concern for closed eyes, eyes struggling to stay open, or falling asleep. "
                            "NEVER provide only the concern level - always include the detailed description."
                        )
                    },
                    {
                        'role': 'user',
                        'content': query,
                        'images': image_list,
                    }
                ]
            )
            response_content = res['message']['content']
            # Log the raw response immediately after receiving it (INFO level for visibility)
            logger.info("="*80)
            logger.info("LLAVA MODEL RESPONSE:")
            logger.info("-"*80)
            logger.info(response_content)
            logger.info("-"*80)
            
            # Warn if response is suspiciously short (likely incomplete)
            if len(response_content.strip()) < 50:
                logger.warning(f"WARNING: Response is very short ({len(response_content)} chars). "
                              f"Expected detailed analysis but got: '{response_content}'")
            
            return response_content
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return None
    
    def _parse_alert_level(self, response: Optional[str]) -> str:
        """
        Parse alert level from LLM response.
        
        Args:
            response: LLM response text (may be None if query failed)
        
        Returns:
            Alert level: 'LOW', 'MODERATE', or 'HIGH'
        """
        if response is None:
            logger.warning("Response is None, defaulting to LOW alert level")
            return 'LOW'  # Default to LOW to avoid false positives
        
        response_upper = response.upper()
        logger.info(f"Parsing alert level from response (length: {len(response)} chars)")
        
        # Check if response starts with explicit level (most reliable)
        first_line = response_upper.split('\n')[0].strip()
        logger.info(f"First line of response: '{first_line[:100]}...'")
        
        if first_line.startswith('HIGH'):
            logger.info("Alert level determined: HIGH (from first line)")
            return 'HIGH'
        elif first_line.startswith('MODERATE'):
            logger.info("Alert level determined: MODERATE (from first line)")
            return 'MODERATE'
        elif first_line.startswith('LOW'):
            logger.info("Alert level determined: LOW (from first line)")
            return 'LOW'
        
        # Check for explicit level mentions
        if 'CONCERN LEVEL: HIGH' in response_upper or 'LEVEL: HIGH' in response_upper:
            logger.info("Alert level determined: HIGH (from explicit level mention)")
            return 'HIGH'
        elif 'CONCERN LEVEL: MODERATE' in response_upper or 'LEVEL: MODERATE' in response_upper:
            logger.info("Alert level determined: MODERATE (from explicit level mention)")
            return 'MODERATE'
        elif 'CONCERN LEVEL: LOW' in response_upper or 'LEVEL: LOW' in response_upper:
            logger.info("Alert level determined: LOW (from explicit level mention)")
            return 'LOW'
        
        # Check for safety-critical keywords (only for HIGH) - focused on closed eyes and severe fatigue
        high_concern_keywords = [
            'EYES CLOSED', 'EYE CLOSED', 'CLOSED EYES', 'EYES SHUT', 'EYE SHUT',
            'HEAVILY DROOPING', 'DROOPING EYES', 'EYES DROOPING', 'HEAVY EYELIDS',
            'STRUGGLING TO STAY OPEN', 'EYES STRUGGLING', 'CANNOT KEEP EYES OPEN',
            'HEAD NODDING', 'NODDING OFF', 'FALLING ASLEEP', 'FALLING ASLEEP',
            'SEVERE DROWSINESS', 'SEVERE FATIGUE', 'EXTREMELY TIRED',
            'DANGEROUS', 'CRITICAL', 'UNRESPONSIVE', 'EYES NOT OPEN'
        ]
        matched_high_keywords = [kw for kw in high_concern_keywords if kw in response_upper]
        if matched_high_keywords:
            logger.info(f"Alert level determined: HIGH (matched keywords: {matched_high_keywords})")
            return 'HIGH'
        
        # Check for moderate concern keywords - fatigue indicators
        moderate_keywords = [
            'MILD DROWSINESS', 'DROWSY', 'SLEEPY', 'SOMEWHAT TIRED',
            'SLIGHTLY DISTRACTED', 'TIRED', 'FATIGUE', 'FATIGUED',
            'EXHAUSTION', 'EXHAUSTED', 'YAWNING', 'YAWN',
            'RUBBING EYES', 'SLOW BLINKING', 'HEAVY EYELIDS',
            'PARTIALLY CLOSED', 'EYES PARTIALLY CLOSED', 'DROWSINESS',
            'DISTRACTED', 'ANGRY', 'SIGNS OF FATIGUE', 'FATIGUE INDICATORS'
        ]
        matched_moderate_keywords = [kw for kw in moderate_keywords if kw in response_upper]
        if matched_moderate_keywords:
            logger.info(f"Alert level determined: MODERATE (matched keywords: {matched_moderate_keywords})")
            return 'MODERATE'
        
        # Check for normal/alert states (LOW)
        low_keywords = ['AWAKE', 'ALERT', 'ATTENTIVE', 'NORMAL', 'NO CONCERN', 'NO ISSUES', 'HEALTHY']
        matched_low_keywords = [kw for kw in low_keywords if kw in response_upper]
        if matched_low_keywords:
            logger.info(f"Alert level determined: LOW (matched keywords: {matched_low_keywords})")
            return 'LOW'
        
        # Default to LOW to avoid false positives
        logger.warning("No alert level keywords matched, defaulting to LOW")
        return 'LOW'
    
    def _trigger_alert(self, analysis_result: Dict[str, any]) -> None:
        """
        Trigger safety alert for concerning behavior.
        
        Args:
            analysis_result: Analysis result dictionary
        """
        alert = {
            'timestamp': analysis_result['timestamp'],
            'type': analysis_result['analysis_type'],
            'level': analysis_result['alert_level'],
            'details': analysis_result['response']
        }
        
        if 'frame_id' in analysis_result:
            alert['frame_id'] = analysis_result['frame_id']
        
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(
            f"SAFETY ALERT - Level: {alert['level']}, "
            f"Time: {alert['timestamp']}"
        )
        
        # Print alert
        print("\n" + "="*60)
        print(f"⚠️  SAFETY ALERT - {alert['level']} CONCERN DETECTED")
        print(f"Time: {alert['timestamp']}")
        print(f"Details: {alert['details'][:200]}...")
        print("="*60 + "\n")


class FastVideoProcessor:
    """
    Fast video processor optimized for speed.
    Minimal overhead, single query per frame sequence.
    """
    
    def __init__(self, config: FastDriverMonitoringConfig, evaluator: Optional[GroundTruthEvaluator] = None, video_filename: Optional[str] = None):
        self.config = config
        self.analyzer = FastDriverAnalyzer(config)
        self.frame_buffer = deque(maxlen=config.FRAME_BUFFER_SIZE)
        self.frame_count = 0
        self.frame_sequence_id = 0
        self.frame_summaries = []
        self.detections_for_evaluation = []  # Store detections for ground truth evaluation
        self.video_fps = None  # Will be set when video is opened
        self.evaluator = evaluator
        self.video_filename = video_filename
    
    def encode_frame(self, frame) -> bytes:
        """
        Encode OpenCV frame to JPEG bytes.
        
        Args:
            frame: OpenCV frame (numpy array)
        
        Returns:
            JPEG encoded frame as bytes
        """
        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode frame")
        return encoded.tobytes()
    
    def frame_generator(self, cap: cv2.VideoCapture, debug_show: bool = False):
        """
        Generator that reads frames from video capture.
        
        Args:
            cap: OpenCV VideoCapture object
            debug_show: Whether to display frames in a window
        
        Yields:
            Tuple of (frame_number, frame)
        """
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from video source")
                break
            
            frame_number += 1
            
            if debug_show:
                # Resize frame to 50% for display
                height, width = frame.shape[:2]
                display_frame = cv2.resize(frame, (width // 2, height // 2))
                cv2.imshow('Fast Driver Monitoring', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested stop")
                    break
            
            yield frame_number, frame
    
    def process_video_stream(
        self,
        cap: cv2.VideoCapture,
        debug_show: bool = False
    ) -> None:
        """
        Process video stream with fast driver monitoring analysis.
        
        Args:
            cap: OpenCV VideoCapture object
            debug_show: Whether to show debug window
        """
        logger.info("Starting fast driver monitoring system...")
        logger.info(f"Processing interval: {self.config.PROCESSING_INTERVAL_SECONDS} seconds")
        logger.info(f"Frame buffer size: {self.config.FRAME_BUFFER_SIZE} frames")
        logger.info("Using single comprehensive query for maximum speed")
        
        # Get video FPS for timestamp calculation
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = self.config.FRAME_RATE
        
        frames_per_interval = self.config.PROCESSING_INTERVAL_SECONDS * self.config.FRAME_RATE
        
        for frame_number, frame in self.frame_generator(cap, debug_show):
            # Add frame to buffer
            frame_bytes = self.encode_frame(frame)
            self.frame_buffer.append(frame_bytes)
            
            # Process at specified intervals
            if frame_number % frames_per_interval == 0 and len(self.frame_buffer) >= 2:
                # Increment frame sequence ID
                self.frame_sequence_id += 1
                
                logger.info(f"Processing frame sequence #{self.frame_sequence_id} at video frame {frame_number}...")
                
                # Convert buffer to list for analysis
                frame_sequence = list(self.frame_buffer)
                
                # Perform single comprehensive analysis
                start_time = time.time()
                results = self.analyzer.analyze_driver_state(frame_sequence)
                total_time = time.time() - start_time
                
                # Add metadata
                results['frame_id'] = self.frame_sequence_id
                results['video_frame_number'] = frame_number
                results['total_processing_time_seconds'] = total_time
                
                # Calculate timestamp for this frame
                timestamp = frame_number / self.video_fps if self.video_fps > 0 else 0
                results['timestamp_seconds'] = timestamp
                
                # Store detection for ground truth evaluation
                detection_record = {
                    'video_frame_number': frame_number,
                    'timestamp': timestamp,
                    'alert_level': results.get('alert_level', 'LOW')
                }
                self.detections_for_evaluation.append(detection_record)
                
                # Store summary
                summary = self._create_frame_summary(results)
                self.frame_summaries.append({
                    'frame_id': self.frame_sequence_id,
                    'video_frame_number': frame_number,
                    'timestamp': results['timestamp'],
                    'summary': summary,
                    'processing_time': total_time,
                    'query_time': results.get('query_time_seconds', 0)
                })
                
                # Print results
                self._print_analysis_results(results)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.05)  # Reduced delay for faster processing
        
        logger.info("Video processing stopped")
    
    def _print_analysis_results(self, results: Dict[str, any]) -> None:
        """
        Print formatted analysis results.
        
        Args:
            results: Analysis results dictionary
        """
        if not results:
            return
        
        frame_id = results.get('frame_id', 'N/A')
        video_frame = results.get('video_frame_number', 'N/A')
        processing_time = results.get('total_processing_time_seconds', 0)
        query_time = results.get('query_time_seconds', 0)
        
        # Get ground truth status if evaluator is available
        ground_truth_status = None
        if self.evaluator and self.video_filename:
            timestamp = results.get('timestamp_seconds', 0)
            is_impaired = self.evaluator.is_impaired_at_time(self.video_filename, timestamp)
            ground_truth_status = "IMPAIRED" if is_impaired else "NORMAL"
        
        print("\n" + "-"*60)
        print(f"Frame #{frame_id} (Video Frame {video_frame}) - {results['timestamp']}")
        print(f"Processing time: {processing_time:.2f}s (Query: {query_time:.2f}s)")
        print("-"*60)
        alert_level = results.get('alert_level', 'N/A')
        if ground_truth_status:
            print(f"Alert Level: {alert_level} | Ground Truth: {ground_truth_status}")
        else:
            print(f"Alert Level: {alert_level}")
        print(f"Response: {results.get('response', 'N/A')}")
        print("-"*60 + "\n")
    
    def _create_frame_summary(self, results: Dict[str, any]) -> str:
        """
        Create a concise summary for a frame analysis.
        
        Args:
            results: Analysis results dictionary
        
        Returns:
            Summary string
        """
        if not results:
            return "No analysis available"
        
        alert_level = results.get('alert_level', 'N/A')
        response = results.get('response', '')
        
        # Extract first sentence or first 100 chars
        if response:
            brief = response.split('.')[0][:100]
            return f"Alert: {alert_level} | {brief}"
        
        return f"Alert: {alert_level}"


def find_video_files(data_dir: str = "data") -> List[str]:
    """
    Find all video files in the data directory (recursively).
    
    Args:
        data_dir: Path to data directory (default: "data")
    
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_files = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return video_files
    
    # Recursively search for video files
    for video_file in data_path.rglob('*'):
        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
            video_files.append(str(video_file))
    
    # Sort for consistent processing order
    video_files.sort()
    
    return video_files


def process_single_video(video_path: str, config: FastDriverMonitoringConfig, debug_show: bool = False) -> Dict[str, any]:
    """
    Process a single video file.
    
    Args:
        video_path: Path to video file
        config: Fast driver monitoring configuration
        debug_show: Whether to show debug window
    
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing video (fast mode): {video_path}")
    
    # Initialize ground truth evaluator
    evaluator = GroundTruthEvaluator()
    
    # Get video filename for ground truth lookup
    video_filename = Path(video_path).name
    
    # Initialize video processor with evaluator and video filename
    processor = FastVideoProcessor(config, evaluator=evaluator, video_filename=video_filename)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return {
            'video_path': video_path,
            'success': False,
            'error': f"Cannot open video file: {video_path}"
        }
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
    
    try:
        # Process video stream
        start_time = time.time()
        processor.process_video_stream(cap, debug_show=debug_show)
        total_processing_time = time.time() - start_time
        
        # Calculate performance metrics
        avg_processing_time = None
        avg_query_time = None
        if processor.frame_summaries:
            processing_times = [s.get('processing_time', 0) for s in processor.frame_summaries if 'processing_time' in s]
            query_times = [s.get('query_time', 0) for s in processor.frame_summaries if 'query_time' in s]
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
            if query_times:
                avg_query_time = sum(query_times) / len(query_times)
        
        # Calculate accuracy if ground truth is available
        accuracy_metrics = None
        if processor.detections_for_evaluation:
            accuracy_metrics = evaluator.calculate_frame_level_accuracy(
                video_filename,
                processor.detections_for_evaluation,
                fps
            )
            if accuracy_metrics.get('has_ground_truth', False):
                evaluator.print_accuracy_report(accuracy_metrics)
                # Save evaluation results to file
                evaluator.save_evaluation_results(accuracy_metrics, video_path)
        
        return {
            'video_path': video_path,
            'success': True,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'frame_summaries': processor.frame_summaries,
            'alerts': processor.analyzer.alert_history,
            'total_analyses': len(processor.frame_summaries),
            'total_alerts': len(processor.analyzer.alert_history),
            'total_processing_time': total_processing_time,
            'avg_analysis_time': avg_processing_time,
            'avg_query_time': avg_query_time,
            'speedup_estimate': '5x faster (single query vs 5 separate analyses)',
            'accuracy_metrics': accuracy_metrics
        }
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return {
            'video_path': video_path,
            'success': False,
            'error': 'Interrupted by user'
        }
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
        return {
            'video_path': video_path,
            'success': False,
            'error': str(e)
        }
    finally:
        # Cleanup
        cap.release()
        if debug_show:
            cv2.destroyAllWindows()


def process_camera_feed(config: FastDriverMonitoringConfig, camera_index: int = 0, debug_show: bool = True) -> None:
    """
    Process live camera feed for fast driver monitoring.
    
    Args:
        config: Fast driver monitoring configuration
        camera_index: Camera device index (default: 0)
        debug_show: Whether to show debug window
    """
    logger.info("Starting fast driver monitoring system with camera feed...")
    
    # Initialize video processor
    processor = FastVideoProcessor(config)
    
    # Open camera feed
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        logger.error(f"Cannot open camera device: {camera_index}")
        raise IOError(f"Cannot open camera device: {camera_index}")
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        logger.info(f"Camera FPS: {fps}")
    else:
        logger.info(f"Using configured FPS: {config.FRAME_RATE}")
    
    try:
        # Process video stream
        processor.process_video_stream(cap, debug_show=debug_show)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("System shutdown complete")
        
        # Print frame summaries
        if processor.frame_summaries:
            print("\n" + "="*60)
            print("FRAME ANALYSIS SUMMARY")
            print("="*60)
            print(f"Total frames analyzed: {len(processor.frame_summaries)}")
            if processor.frame_summaries:
                avg_time = sum(s.get('processing_time', 0) for s in processor.frame_summaries) / len(processor.frame_summaries)
                print(f"Average processing time: {avg_time:.2f}s per frame sequence")
            print("-"*60)
            for frame_summary in processor.frame_summaries:
                print(f"\nFrame ID: {frame_summary['frame_id']} (Video Frame: {frame_summary['video_frame_number']})")
                print(f"Timestamp: {frame_summary['timestamp']}")
                print(f"Processing time: {frame_summary.get('processing_time', 0):.2f}s")
                print(f"Summary: {frame_summary['summary']}")
            print("="*60)
        
        # Print alert summary
        if processor.analyzer.alert_history:
            print("\n" + "="*60)
            print("ALERT SUMMARY")
            print("="*60)
            for alert in processor.analyzer.alert_history:
                frame_id_str = f"Frame #{alert.get('frame_id', 'N/A')} - " if 'frame_id' in alert else ""
                print(f"[{alert['timestamp']}] {frame_id_str}{alert['level']}")
            print("="*60)


def main():
    """
    Main function to run fast in-car driver monitoring system.
    Supports both video file processing and live camera feed.
    """
    parser = argparse.ArgumentParser(
        description='Fast driver monitoring system - single query for maximum speed'
    )
    parser.add_argument(
        '--camera',
        action='store_true',
        help='Use live camera feed instead of video files (default: process videos from data folder)'
    )
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Camera device index (default: 0, only used with --camera)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory containing videos (default: data, only used without --camera)'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Process a specific video file (if not specified, processes all videos in data directory, only used without --camera)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display window (faster processing)'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = FastDriverMonitoringConfig()
    debug_show = not args.no_display
    
    # Process camera feed if requested
    if args.camera:
        process_camera_feed(config, camera_index=args.camera_index, debug_show=debug_show)
        return
    
    # Otherwise, process video files
    # Determine which videos to process
    if args.video:
        # Process single specified video
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {args.video}")
            return
        video_files = [str(video_path)]
    else:
        # Find all videos in data directory
        video_files = find_video_files(args.data_dir)
        if not video_files:
            logger.error(f"No video files found in {args.data_dir}")
            return
    
    logger.info(f"Found {len(video_files)} video file(s) to process")
    logger.info("Using fast mode: single comprehensive query (5x faster than standard)")
    
    # Process each video
    all_results = []
    
    for i, video_path in enumerate(video_files, 1):
        print("\n" + "="*80)
        print(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print("="*80)
        
        result = process_single_video(video_path, config, debug_show=debug_show)
        all_results.append(result)
        
        # Print per-video summary
        if result['success']:
            print(f"\n✓ Completed: {os.path.basename(video_path)}")
            print(f"  - Total analyses: {result['total_analyses']}")
            print(f"  - Total alerts: {result['total_alerts']}")
            if result.get('avg_analysis_time'):
                print(f"  - Avg analysis time: {result['avg_analysis_time']:.2f}s")
            if result.get('avg_query_time'):
                print(f"  - Avg query time: {result['avg_query_time']:.2f}s")
            if result.get('speedup_estimate'):
                print(f"  - Performance: {result['speedup_estimate']}")
            
            if result['frame_summaries']:
                print("\n  Frame Analysis Summary:")
                for frame_summary in result['frame_summaries']:
                    print(f"    Frame ID: {frame_summary['frame_id']} - {frame_summary['summary']}")
            
            if result['alerts']:
                print("\n  Alerts:")
                for alert in result['alerts']:
                    frame_id_str = f"Frame #{alert.get('frame_id', 'N/A')} - " if 'frame_id' in alert else ""
                    print(f"    [{alert['level']}] {frame_id_str}{alert['type']}")
        else:
            print(f"\n✗ Failed: {os.path.basename(video_path)} - {result.get('error', 'Unknown error')}")
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL PROCESSING SUMMARY")
    print("="*80)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"Total videos processed: {len(all_results)}")
    print(f"  - Successful: {len(successful)}")
    print(f"  - Failed: {len(failed)}")
    
    if successful:
        total_analyses = sum(r['total_analyses'] for r in successful)
        total_alerts = sum(r['total_alerts'] for r in successful)
        print(f"\nTotal analyses across all videos: {total_analyses}")
        print(f"Total alerts across all videos: {total_alerts}")
        
        # Performance summary
        avg_times = [r.get('avg_analysis_time') for r in successful if r.get('avg_analysis_time')]
        avg_query_times = [r.get('avg_query_time') for r in successful if r.get('avg_query_time')]
        if avg_times:
            overall_avg = sum(avg_times) / len(avg_times)
            print(f"\nAverage analysis time per frame sequence: {overall_avg:.2f}s")
        if avg_query_times:
            overall_query_avg = sum(avg_query_times) / len(avg_query_times)
            print(f"Average query time: {overall_query_avg:.2f}s")
        print("Speedup: ~5x faster than standard version (single query vs 5 separate analyses)")
        
        # List all alerts from all videos
        if total_alerts > 0:
            print("\n" + "-"*80)
            print("ALL ALERTS ACROSS ALL VIDEOS")
            print("-"*80)
            for result in successful:
                if result['alerts']:
                    print(f"\nVideo: {os.path.basename(result['video_path'])}")
                    for alert in result['alerts']:
                        frame_id_str = f"Frame #{alert.get('frame_id', 'N/A')} - " if 'frame_id' in alert else ""
                        print(f"  [{alert['timestamp']}] {frame_id_str}{alert['level']} - {alert['type']}")
    
    if failed:
        print("\n" + "-"*80)
        print("FAILED VIDEOS")
        print("-"*80)
        for result in failed:
            print(f"  {os.path.basename(result['video_path'])}: {result.get('error', 'Unknown error')}")
    
    print("="*80)
    logger.info("All videos processed")


if __name__ == "__main__":
    main()

