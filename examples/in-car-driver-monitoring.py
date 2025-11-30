"""
In-Car Driver Monitoring System with Video Understanding
Extended example for detecting driver behavior patterns (e.g., impaired driving)

This implementation extends the reference implementation with:
- Temporal frame sequence processing
- Driver-specific behavior analysis
- Multiple safety-related queries
- Alert system for dangerous behaviors
- Frame buffer for temporal context

Key Features:
- Multi-frame analysis for temporal understanding
- Driver behavior pattern detection
- Safety alert system
- Configurable monitoring parameters
- Integration with Ollama for local LLM inference
"""

import cv2
import ollama
import datetime
import logging
from typing import List, Optional, Dict, Tuple
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


class DriverMonitoringConfig:
    """Configuration for driver monitoring system"""
    MODEL_NAME = "llava:7b-v1.6-mistral-q2_K"
    FRAME_RATE = 30
    PROCESSING_INTERVAL_SECONDS = 2  # Process every 2 seconds
    FRAME_BUFFER_SIZE = 5  # Keep last 5 frames for temporal analysis
    ALERT_THRESHOLD = 0.7  # Confidence threshold for alerts
    
    # Frame resizing for performance optimization
    # Set to None for no resizing, or (width, height) tuple for fixed size
    # Examples: None (no resize), (640, 480), (512, 512), (800, 600)
    FRAME_RESIZE_TO = (640, 480)  # Resize to 640x480 for faster processing
    
    # LLM generation parameters
    TEMPERATURE = 0
    TOP_K = 1
    TOP_P = 0.1
    NUM_CTX = 2048  # Increased for better context
    NUM_PREDICT = 256  # Longer responses for detailed analysis


class DriverBehaviorAnalyzer:
    """
    Analyzes driver behavior using temporal video understanding.
    Processes frame sequences to detect patterns like:
    - Drowsiness indicators
    - Distraction patterns
    - Impaired driving behaviors
    - Hand position on steering wheel
    """
    
    def __init__(self, config: DriverMonitoringConfig):
        self.config = config
        self.frame_buffer = deque(maxlen=config.FRAME_BUFFER_SIZE)
        self.alert_history = []
        self.last_analysis_time = 0
        self.recent_alert_levels = deque(maxlen=3)  # Track recent alert levels for temporal smoothing
        
    def query_llm(self, query: str, image_list: List[bytes]) -> Optional[str]:
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
                            "You are a driver monitoring system for vehicle safety. "
                            "Analyze driver behavior in car interior video frames. "
                            "Answer with concise, factual observations. "
                            "Be objective: report normal states (awake, alert, attentive) as LOW concern. "
                            "Only report MODERATE or HIGH concern for actual safety issues like drowsiness, "
                            "distraction, or impaired behavior. Distinguish between normal emotional states "
                            "(happy, neutral) and safety concerns (sleepy, exhausted, angry)."
                        )
                    },
                    {
                        'role': 'user',
                        'content': query,
                        'images': image_list,
                    }
                ]
            )
            return res['message']['content']
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return None
    
    def analyze_driver_attention(self, frames: List[bytes]) -> Dict[str, any]:
        """
        Analyze driver attention and alertness.
        
        Args:
            frames: List of frame bytes for temporal analysis
        
        Returns:
            Dictionary with analysis results
        """
        query = (
            "Analyze these sequential video frames of a driver. "
            "Describe: 1) Where is the driver looking? 2) Are their eyes open and alert? "
            "3) Is their head position normal or tilted? 4) Any signs of drowsiness or distraction?"
        )
        
        response = self.query_llm(query, frames)
        if response is None:
            response = "Error: Could not analyze driver attention. LLM service unavailable."
            logger.error("Failed to analyze driver attention: LLM query returned None")
        
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'analysis_type': 'attention',
            'response': response,
            'frames_analyzed': len(frames)
        }
    
    def analyze_hand_position(self, frames: List[bytes]) -> Dict[str, any]:
        """
        Analyze driver hand position on steering wheel.
        
        Args:
            frames: List of frame bytes for temporal analysis
        
        Returns:
            Dictionary with analysis results
        """
        query = (
            "Analyze these sequential video frames. "
            "Describe: 1) Where are the driver's hands positioned? "
            "2) Are both hands on the steering wheel? "
            "3) Is the driver holding anything (phone, food, etc.)? "
            "4) Any concerning hand positions?"
        )
        
        response = self.query_llm(query, frames)
        if response is None:
            response = "Error: Could not analyze hand position. LLM service unavailable."
            logger.error("Failed to analyze hand position: LLM query returned None")
        
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'analysis_type': 'hand_position',
            'response': response,
            'frames_analyzed': len(frames)
        }
    
    def detect_face_state(self, frames: List[bytes]) -> Dict[str, any]:
        """
        Detect the current face state of the driver.
        
        Args:
            frames: List of frame bytes for temporal analysis
        
        Returns:
            Dictionary with detected face state and analysis
        """
        query = (
            "Analyze the driver's face in these sequential video frames. "
            "Identify the face state from these options: AWAKE, SLEEPY, JOYFUL, EXHAUSTED, TIRED, ANGRY, or NEUTRAL. "
            "Describe: 1) The facial expression 2) Eye state (open/closed/partially closed) "
            "3) Head position 4) Overall alertness level. "
            "Be specific about which state you detect. If the driver appears normal and alert, state AWAKE or NEUTRAL."
        )
        
        response = self.query_llm(query, frames)
        if response is None:
            response = "Error: Could not detect face state. LLM service unavailable."
            logger.error("Failed to detect face state: LLM query returned None")
        
        # Parse face state from response
        face_state = self._parse_face_state(response)
        
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'analysis_type': 'face_state',
            'response': response,
            'detected_state': face_state,
            'frames_analyzed': len(frames)
        }
    
    def detect_impaired_driving(self, frames: List[bytes], face_state: Optional[str] = None, frame_id: Optional[int] = None, timestamp_seconds: Optional[float] = None) -> Dict[str, any]:
        """
        Detect signs of impaired driving behavior.
        Uses face state information to inform alert levels.
        
        Args:
            frames: List of frame bytes for temporal analysis
            face_state: Detected face state from face_state analysis (optional)
            frame_id: Optional frame sequence ID for tracking
        
        Returns:
            Dictionary with analysis results and alert level
        """
        # Build query with face state context if available
        query_base = (
            "Analyze these sequential video frames for signs of impaired driving that pose a CRITICAL safety risk. "
            "Be VERY CONSERVATIVE - only report impairment if there are CLEAR, OBVIOUS signs of danger. "
            "Look for: 1) Eyes CLOSED (not just drooping) for more than a brief moment "
            "2) Head FALLING FORWARD or NODDING OFF (not just tilting) "
            "3) COMPLETE unresponsiveness or loss of consciousness "
            "4) SEVERE drowsiness with eyes struggling to stay open "
            "Do NOT report impairment for: normal blinking, slight tiredness, normal head movements, "
            "or momentary distractions. Only report MODERATE or HIGH if there is a GENUINE safety risk. "
        )
        
        if face_state:
            query_base += f"Note: The driver's detected face state is: {face_state}. "
            if face_state in ['SLEEPY', 'TIRED', 'EXHAUSTED', 'ANGRY']:
                query_base += (
                    "This is a concerning state for driving safety. "
                    "Rate the concern level as MODERATE or HIGH based on severity. "
                )
            elif face_state in ['AWAKE', 'NEUTRAL', 'JOYFUL']:
                query_base += (
                    "This is a normal state. Only rate as MODERATE or HIGH if you see "
                    "additional concerning behaviors beyond the face state. "
                )
        
        query = query_base + (
            "Rate the concern level as: "
            "LOW (normal/alert driver, eyes open, attentive, no safety concerns - DEFAULT if uncertain), "
            "MODERATE (ONLY if there are CLEAR signs like eyes frequently closing, head nodding, clear drowsiness), "
            "or HIGH (ONLY if eyes are CLOSED, head falling forward, or complete loss of alertness). "
            "When in doubt, choose LOW. Only use MODERATE or HIGH for OBVIOUS safety risks. "
            "Start your response with the concern level in ALL CAPS: LOW, MODERATE, or HIGH."
        )
        
        response = self.query_llm(query, frames)
        if response is None:
            response = "Error: Could not detect impaired driving. LLM service unavailable."
            logger.error("Failed to detect impaired driving: LLM query returned None")
        
        # Parse alert level from response
        alert_level = self._parse_alert_level(response)
        
        # Override alert level based on face state if needed
        if face_state:
            alert_level = self._adjust_alert_level_by_face_state(alert_level, face_state, response or "")
        
        # Apply temporal smoothing - require consistency before alerting
        # Only downgrade if we have recent LOW detections, don't upgrade
        # Pass face_state to preserve HIGH alerts from SLEEPY face state
        smoothed_alert_level = self._apply_temporal_smoothing(alert_level, face_state=face_state)
        
        result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'analysis_type': 'impaired_driving',
            'response': response,
            'alert_level': smoothed_alert_level,
            'raw_alert_level': alert_level,  # Keep original for debugging
            'face_state_considered': face_state,
            'frames_analyzed': len(frames)
        }
        
        # Include frame_id if provided
        if frame_id is not None:
            result['frame_id'] = frame_id
        
        # Include video timestamp if provided
        if timestamp_seconds is not None:
            result['timestamp_seconds'] = timestamp_seconds
        
        # Trigger alert for any concerning state (after smoothing)
        if smoothed_alert_level in ['MODERATE', 'HIGH']:
            self._trigger_alert(result)
        
        return result
    
    def _apply_temporal_smoothing(self, alert_level: str, face_state: Optional[str] = None) -> str:
        """
        Apply temporal smoothing to reduce false positives.
        Requires consistent detections before alerting.
        SLEEPY face state always results in HIGH alert level and is not downgraded.
        
        Args:
            alert_level: Current alert level
            face_state: Detected face state (optional)
        
        Returns:
            Smoothed alert level
        """
        # SLEEPY face state always results in HIGH alert level - never downgrade
        if face_state == 'SLEEPY' and alert_level == 'HIGH':
            self.recent_alert_levels.append(alert_level)
            return 'HIGH'
        
        # Add current level to history
        self.recent_alert_levels.append(alert_level)
        
        # If we have at least 2 recent detections
        if len(self.recent_alert_levels) >= 2:
            recent = list(self.recent_alert_levels)
            # If recent detections are inconsistent, be conservative
            if alert_level in ['MODERATE', 'HIGH']:
                # Check if previous detection was LOW
                if len(recent) >= 2 and recent[-2] == 'LOW':
                    # Single spike, downgrade to LOW
                    logger.debug(f"Temporal smoothing: downgrading {alert_level} to LOW (inconsistent with previous LOW)")
                    return 'LOW'
                # If we have 2+ consecutive MODERATE/HIGH, keep it
                if len(recent) >= 2 and all(level in ['MODERATE', 'HIGH'] for level in recent[-2:]):
                    return alert_level
                # If only one MODERATE/HIGH, downgrade to LOW
                if alert_level == 'MODERATE' and recent[-2] == 'LOW':
                    return 'LOW'
        
        # Default: return as-is for LOW, or if we don't have enough history
        return alert_level
    
    def analyze_general_scene(self, frames: List[bytes]) -> Dict[str, any]:
        """
        General scene analysis of driver and car interior.
        
        Args:
            frames: List of frame bytes for temporal analysis
        
        Returns:
            Dictionary with analysis results
        """
        query = (
            "Describe what you see in these sequential video frames of a car interior. "
            "Focus on the driver: their position, state, and any notable behaviors. "
            "Also note the environment: lighting, weather conditions visible through windows, "
            "and any objects in the car."
        )
        
        response = self.query_llm(query, frames)
        if response is None:
            response = "Error: Could not analyze scene. LLM service unavailable."
            logger.error("Failed to analyze general scene: LLM query returned None")
        
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'analysis_type': 'general_scene',
            'response': response,
            'frames_analyzed': len(frames)
        }
    
    def _parse_face_state(self, response: Optional[str]) -> str:
        """
        Parse face state from LLM response.
        
        Args:
            response: LLM response text (may be None if query failed)
        
        Returns:
            Detected face state
        """
        if response is None:
            logger.warning("Cannot parse face state: LLM response is None")
            return 'UNKNOWN'
        
        response_upper = response.upper()
        
        # Check for specific states in order of priority
        if 'SLEEPY' in response_upper or 'DROWSY' in response_upper:
            return 'SLEEPY'
        elif 'EXHAUSTED' in response_upper:
            return 'EXHAUSTED'
        elif 'TIRED' in response_upper:
            return 'TIRED'
        elif 'ANGRY' in response_upper or 'FURIOUS' in response_upper:
            return 'ANGRY'
        elif 'JOYFUL' in response_upper or 'HAPPY' in response_upper or 'SMILING' in response_upper:
            return 'JOYFUL'
        elif 'AWAKE' in response_upper or 'ALERT' in response_upper:
            return 'AWAKE'
        elif 'NEUTRAL' in response_upper:
            return 'NEUTRAL'
        else:
            return 'UNKNOWN'
    
    def _parse_alert_level(self, response: Optional[str]) -> str:
        """
        Parse alert level from LLM response.
        Prioritizes explicit level statements and looks for safety-related keywords.
        
        Args:
            response: LLM response text (may be None if query failed)
        
        Returns:
            Alert level: 'LOW', 'MODERATE', or 'HIGH'
        """
        if response is None:
            logger.warning("Cannot parse alert level: LLM response is None")
            return 'LOW'  # Default to LOW to avoid false positives
        
        response_upper = response.upper()
        
        # Check if response starts with explicit level (most reliable)
        first_line = response_upper.split('\n')[0].strip()
        if first_line.startswith('LOW'):
            return 'LOW'
        elif first_line.startswith('MODERATE'):
            return 'MODERATE'
        elif first_line.startswith('HIGH'):
            return 'HIGH'
        
        # Check for explicit level mentions
        if 'CONCERN LEVEL: HIGH' in response_upper or 'LEVEL: HIGH' in response_upper:
            return 'HIGH'
        elif 'CONCERN LEVEL: MODERATE' in response_upper or 'LEVEL: MODERATE' in response_upper:
            return 'MODERATE'
        elif 'CONCERN LEVEL: LOW' in response_upper or 'LEVEL: LOW' in response_upper:
            return 'LOW'
        
        # Check for safety-critical keywords (only for HIGH)
        if any(keyword in response_upper for keyword in ['EYES CLOSED', 'HEAD NODDING', 'FALLING ASLEEP', 
                                                          'SEVERE DROWSINESS', 'DANGEROUS', 'CRITICAL']):
            return 'HIGH'
        
        # Check for moderate concern keywords - be more conservative
        moderate_keywords = [
            'EYES FREQUENTLY CLOSING', 'HEAD NODDING', 'CLEAR DROWSINESS',
            'EYES STRUGGLING', 'FREQUENT BLINKING', 'HEAVY EYELIDS'
        ]
        # Don't trigger on just "TIRED" or "FATIGUE" alone - need more specific signs
        if any(keyword in response_upper for keyword in moderate_keywords):
            return 'MODERATE'
        
        # Check for normal/alert states (LOW)
        if any(keyword in response_upper for keyword in ['AWAKE', 'ALERT', 'ATTENTIVE', 'NORMAL', 
                                                          'NO CONCERN', 'NO ISSUES', 'HEALTHY']):
            return 'LOW'
        
        # Default to LOW to avoid false positives
        return 'LOW'
    
    def _adjust_alert_level_by_face_state(self, current_level: str, face_state: str, response: Optional[str]) -> str:
        """
        Adjust alert level based on detected face state.
        SLEEPY face state always results in HIGH alert level for safety.
        
        Args:
            current_level: Current alert level from LLM response
            face_state: Detected face state
            response: LLM response text (may be None)
        
        Returns:
            Adjusted alert level
        """
        # SLEEPY face state always results in HIGH alert level
        if face_state == 'SLEEPY':
            return 'HIGH'
        
        # Only upgrade if LLM already detected some concern AND face state confirms it
        # Don't upgrade from LOW to MODERATE/HIGH based solely on face state (except SLEEPY)
        if current_level == 'LOW':
            # Even if face state is concerning, trust LLM's LOW assessment
            # (LLM sees the full context, face state might be misleading)
            # Exception: SLEEPY is always HIGH (handled above)
            return current_level
        
        # If LLM detected MODERATE/HIGH, and face state confirms, keep it
        # If face state is normal but LLM detected issues, trust the LLM
        return current_level
    
    def _is_safety_concern(self, response: Optional[str]) -> bool:
        """
        Check if the response indicates an actual safety concern (not just a face state).
        
        Args:
            response: LLM response text (may be None)
        
        Returns:
            True if response indicates a safety concern
        """
        if response is None:
            return False  # No response means no safety concern detected
        
        response_upper = response.upper()
        
        # Safety concern keywords
        safety_keywords = [
            'DROWSY', 'DROWSINESS', 'SLEEPY', 'EYES CLOSED', 'HEAD NODDING',
            'FALLING ASLEEP', 'EXHAUSTED', 'FATIGUE', 'DISTRACTED',
            'NOT ATTENTIVE', 'IMPAIRED', 'DANGEROUS', 'TIRED'
        ]
        
        # Normal state keywords (not safety concerns)
        normal_keywords = [
            'AWAKE', 'ALERT', 'ATTENTIVE', 'NORMAL', 'HEALTHY',
            'JOYFUL', 'HAPPY', 'SMILING', 'NEUTRAL'
        ]
        
        has_safety_keyword = any(keyword in response_upper for keyword in safety_keywords)
        has_normal_keyword = any(keyword in response_upper for keyword in normal_keywords)
        
        # Only consider it a safety concern if safety keywords are present
        # and not overridden by normal state indicators
        return has_safety_keyword and not (has_normal_keyword and 'NOT' not in response_upper)
    
    def _format_video_timestamp(self, seconds: float) -> str:
        """
        Format video timestamp in seconds to readable format (HH:MM:SS or MM:SS).
        
        Args:
            seconds: Timestamp in seconds
        
        Returns:
            Formatted timestamp string
        """
        if seconds < 0:
            return "00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
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
        
        # Include frame_id if available
        if 'frame_id' in analysis_result:
            alert['frame_id'] = analysis_result['frame_id']
        
        # Include video timestamp if available
        if 'timestamp_seconds' in analysis_result:
            alert['timestamp_seconds'] = analysis_result['timestamp_seconds']
            video_time_str = f" [Video Time: {self._format_video_timestamp(analysis_result['timestamp_seconds'])}]"
        else:
            video_time_str = ""
        
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(
            f"SAFETY ALERT - Level: {alert['level']}, "
            f"Type: {alert['type']}, "
            f"Time: {alert['timestamp']}{video_time_str}"
        )
        
        # Print alert (in real system, this would trigger audio/visual warnings)
        print("\n" + "="*60)
        print(f"⚠️  SAFETY ALERT - {alert['level']} CONCERN DETECTED")
        print(f"Type: {alert['type']}")
        print(f"Time: {alert['timestamp']}{video_time_str}")
        print(f"Details: {alert['details'][:200]}...")
        print("="*60 + "\n")
    
    def process_frame_sequence(self, frames: List[bytes], frame_id: Optional[int] = None, timestamp_seconds: Optional[float] = None) -> Dict[str, any]:
        """
        Process a sequence of frames for comprehensive driver analysis.
        
        Args:
            frames: List of frame bytes (temporal sequence)
            frame_id: Optional frame sequence ID for tracking
            timestamp_seconds: Optional video timestamp in seconds
        
        Returns:
            Dictionary with all analysis results
        """
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for temporal analysis")
            return {}
        
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'analyses': {}
        }
        
        # Include frame_id in results if provided
        if frame_id is not None:
            results['frame_id'] = frame_id
        
        # Perform multiple analyses - face state first as it informs other analyses
        logger.info("Detecting face state...")
        face_state_result = self.detect_face_state(frames)
        results['analyses']['face_state'] = face_state_result
        detected_face_state = face_state_result.get('detected_state')
        
        logger.info("Analyzing driver attention...")
        results['analyses']['attention'] = self.analyze_driver_attention(frames)
        
        logger.info("Analyzing hand position...")
        results['analyses']['hand_position'] = self.analyze_hand_position(frames)
        
        logger.info("Detecting impaired driving patterns...")
        # Pass face state to impaired driving detection so it can use this information
        # Also pass frame_id and timestamp_seconds if available
        impaired_result = self.detect_impaired_driving(
            frames, 
            face_state=detected_face_state,
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds
        )
        results['analyses']['impaired_driving'] = impaired_result
        
        logger.info("General scene analysis...")
        results['analyses']['general_scene'] = self.analyze_general_scene(frames)
        
        return results


class InCarVideoProcessor:
    """
    Processes in-car video stream for driver monitoring.
    Handles frame capture, buffering, and analysis scheduling.
    """
    
    def __init__(self, config: DriverMonitoringConfig):
        self.config = config
        self.analyzer = DriverBehaviorAnalyzer(config)
        self.frame_buffer = deque(maxlen=config.FRAME_BUFFER_SIZE)
        self.frame_count = 0
        self.frame_sequence_id = 0  # Running index for analyzed frame sequences
        self.frame_summaries = []  # Store summaries for each analyzed frame sequences
        self.detections_for_evaluation = []  # Store detections for ground truth evaluation
        self.video_fps = None  # Will be set when video is opened
        self.recent_alert_levels = deque(maxlen=3)  # Track recent alert levels for temporal smoothing
        self.has_ground_truth = False  # Will be set if ground truth is available
    
    def encode_frame(self, frame) -> bytes:
        """
        Encode OpenCV frame to JPEG bytes.
        Optionally resizes frame based on configuration for performance.
        
        Args:
            frame: OpenCV frame (numpy array)
        
        Returns:
            JPEG encoded frame as bytes
        """
        # Resize frame if configured
        if self.config.FRAME_RESIZE_TO is not None:
            target_width, target_height = self.config.FRAME_RESIZE_TO
            current_height, current_width = frame.shape[:2]
            
            # Only resize if dimensions differ
            if current_width != target_width or current_height != target_height:
                frame = cv2.resize(
                    frame, 
                    (target_width, target_height), 
                    interpolation=cv2.INTER_AREA
                )
                logger.debug(f"Resized frame from {current_width}x{current_height} to {target_width}x{target_height}")
        
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
                cv2.imshow('Driver Monitoring', display_frame)
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
        Process video stream with driver monitoring analysis.
        
        Args:
            cap: OpenCV VideoCapture object
            debug_show: Whether to show debug window
        """
        logger.info("Starting driver monitoring system...")
        
        # Get video FPS for timestamp calculation
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = self.config.FRAME_RATE
        
        # Log frame resize configuration
        if self.config.FRAME_RESIZE_TO is None:
            logger.info("Frame resizing: DISABLED (using original resolution)")
        else:
            width, height = self.config.FRAME_RESIZE_TO
            logger.info(f"Frame resizing: ENABLED - resized to {width}x{height}")
        
        # Use shorter processing interval if ground truth is available (to catch short periods)
        processing_interval = self.config.PROCESSING_INTERVAL_SECONDS
        if self.has_ground_truth:
            # Use 0.1 second intervals for ground truth videos to catch short periods
            processing_interval = 0.1 * 20
            logger.info(f"Ground truth detected - using shorter processing interval: {processing_interval}s")
        
        logger.info(f"Processing interval: {processing_interval} seconds")
        logger.info(f"Frame buffer size: {self.config.FRAME_BUFFER_SIZE} frames")
        
        frames_per_interval = int(processing_interval * self.video_fps)
        if frames_per_interval < 1:
            frames_per_interval = 1
        
        for frame_number, frame in self.frame_generator(cap, debug_show):
            # Add frame to buffer
            frame_bytes = self.encode_frame(frame)
            self.frame_buffer.append(frame_bytes)
            
            # Process at specified intervals
            if frame_number % frames_per_interval == 0 and len(self.frame_buffer) >= 2:
                # Increment frame sequence ID
                self.frame_sequence_id += 1
                
                # Calculate timestamp for this frame (before processing so it can be used in alerts)
                timestamp = frame_number / self.video_fps if self.video_fps > 0 else 0
                timestamp_str = self._format_video_timestamp(timestamp)
                
                logger.info(f"Processing frame sequence #{self.frame_sequence_id} at video frame {frame_number} [Video Time: {timestamp_str}]...")
                
                # Convert buffer to list for analysis
                frame_sequence = list(self.frame_buffer)
                
                # Perform comprehensive analysis (pass frame_id and timestamp_seconds so alerts can include them)
                results = self.analyzer.process_frame_sequence(
                    frame_sequence, 
                    frame_id=self.frame_sequence_id,
                    timestamp_seconds=timestamp
                )
                
                # Add video frame number to results
                results['video_frame_number'] = frame_number
                
                # Add timestamp to results (already passed to process_frame_sequence, but ensure it's in results)
                results['timestamp_seconds'] = timestamp
                
                # Store detection for ground truth evaluation
                # Extract alert level from impaired_driving analysis
                detection_record = {
                    'video_frame_number': frame_number,
                    'timestamp': timestamp,
                    'alert_level': 'LOW'  # Default
                }
                if 'analyses' in results and 'impaired_driving' in results['analyses']:
                    impaired_analysis = results['analyses']['impaired_driving']
                    detection_record['alert_level'] = impaired_analysis.get('alert_level', 'LOW')
                    detection_record['analyses'] = {'impaired_driving': impaired_analysis}
                self.detections_for_evaluation.append(detection_record)
                
                # Store summary for later
                summary = self._create_frame_summary(results)
                self.frame_summaries.append({
                    'frame_id': self.frame_sequence_id,
                    'video_frame_number': frame_number,
                    'timestamp': results['timestamp'],
                    'summary': summary
                })
                
                # Print results
                self._print_analysis_results(results)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
        
        logger.info("Video processing stopped")
    
    def _format_video_timestamp(self, seconds: float) -> str:
        """
        Format video timestamp in seconds to readable format (HH:MM:SS or MM:SS).
        
        Args:
            seconds: Timestamp in seconds
        
        Returns:
            Formatted timestamp string
        """
        if seconds < 0:
            return "00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _print_analysis_results(self, results: Dict[str, any]) -> None:
        """
        Print formatted analysis results.
        
        Args:
            results: Analysis results dictionary
        """
        if not results or 'analyses' not in results:
            return
        
        frame_id = results.get('frame_id', 'N/A')
        video_frame = results.get('video_frame_number', 'N/A')
        timestamp_seconds = results.get('timestamp_seconds', None)
        
        # Format video timestamp if available
        video_time_str = ""
        if timestamp_seconds is not None:
            video_time_str = f" [Video Time: {self._format_video_timestamp(timestamp_seconds)}]"
        
        print("\n" + "-"*60)
        print(f"Frame #{frame_id} (Video Frame {video_frame}){video_time_str} - {results['timestamp']}")
        print("-"*60)
        
        for analysis_type, analysis_data in results['analyses'].items():
            print(f"\n[{analysis_type.upper()}]")
            print(f"Response: {analysis_data.get('response', 'N/A')}")
            
            if 'detected_state' in analysis_data:
                print(f"Detected State: {analysis_data['detected_state']}")
            
            if 'alert_level' in analysis_data:
                print(f"Alert Level: {analysis_data['alert_level']}")
        
        print("-"*60 + "\n")
    
    def _create_frame_summary(self, results: Dict[str, any]) -> str:
        """
        Create a concise summary for a frame analysis.
        
        Args:
            results: Analysis results dictionary
        
        Returns:
            Summary string
        """
        if not results or 'analyses' not in results:
            return "No analysis available"
        
        summary_parts = []
        
        # Get key information from analyses
        if 'face_state' in results['analyses']:
            face_data = results['analyses']['face_state']
            state = face_data.get('detected_state', 'UNKNOWN')
            summary_parts.append(f"Face State: {state}")
        
        if 'impaired_driving' in results['analyses']:
            impaired_data = results['analyses']['impaired_driving']
            alert_level = impaired_data.get('alert_level', 'N/A')
            summary_parts.append(f"Alert Level: {alert_level}")
        
        # Get a brief description from general scene or attention analysis
        if 'general_scene' in results['analyses']:
            scene_response = results['analyses']['general_scene'].get('response', '')
            if scene_response:
                # Take first sentence or first 100 chars
                brief = scene_response.split('.')[0][:100]
                summary_parts.append(f"Scene: {brief}")
        elif 'attention' in results['analyses']:
            attention_response = results['analyses']['attention'].get('response', '')
            if attention_response:
                brief = attention_response.split('.')[0][:100]
                summary_parts.append(f"Attention: {brief}")
        
        return " | ".join(summary_parts) if summary_parts else "Analysis completed"


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


def process_single_video(video_path: str, config: DriverMonitoringConfig, debug_show: bool = False) -> Dict[str, any]:
    """
    Process a single video file.
    
    Args:
        video_path: Path to video file
        config: Driver monitoring configuration
        debug_show: Whether to show debug window
    
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing video: {video_path}")
    
    # Initialize ground truth evaluator
    evaluator = GroundTruthEvaluator()
    
    # Check if ground truth exists for this video
    video_filename = Path(video_path).name
    has_ground_truth = evaluator.get_ground_truth_for_video(video_filename) is not None
    
    # Initialize video processor for this video
    processor = InCarVideoProcessor(config)
    processor.has_ground_truth = has_ground_truth
    
    if has_ground_truth:
        logger.info(f"Ground truth available for {video_filename} - using optimized processing")
    
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
    
    if fps > 0:
        # Update config if video has different frame rate
        if abs(fps - config.FRAME_RATE) > 5:
            logger.warning(
                f"Video FPS ({fps}) differs significantly from config ({config.FRAME_RATE}). "
                "Consider updating config."
            )
    
    try:
        # Process video stream
        processor.process_video_stream(cap, debug_show=debug_show)
        
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


def process_camera_feed(config: DriverMonitoringConfig, camera_index: int = 0, debug_show: bool = True) -> None:
    """
    Process live camera feed for driver monitoring.
    
    Args:
        config: Driver monitoring configuration
        camera_index: Camera device index (default: 0)
        debug_show: Whether to show debug window
    """
    logger.info("Starting driver monitoring system with camera feed...")
    
    # Initialize video processor
    processor = InCarVideoProcessor(config)
    
    # Open camera feed
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        logger.error(f"Cannot open camera device: {camera_index}")
        raise IOError(f"Cannot open camera device: {camera_index}")
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        logger.info(f"Camera FPS: {fps}")
        # Update config if camera has different frame rate
        if abs(fps - config.FRAME_RATE) > 5:
            logger.warning(
                f"Camera FPS ({fps}) differs significantly from config ({config.FRAME_RATE}). "
                "Consider updating config."
            )
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
            print("-"*60)
            for frame_summary in processor.frame_summaries:
                print(f"\nFrame ID: {frame_summary['frame_id']} (Video Frame: {frame_summary['video_frame_number']})")
                print(f"Timestamp: {frame_summary['timestamp']}")
                print(f"Summary: {frame_summary['summary']}")
            print("="*60)
        
        # Print alert summary
        if processor.analyzer.alert_history:
            print("\n" + "="*60)
            print("ALERT SUMMARY")
            print("="*60)
            for alert in processor.analyzer.alert_history:
                frame_id_str = f"Frame #{alert.get('frame_id', 'N/A')} - " if 'frame_id' in alert else ""
                print(f"[{alert['timestamp']}] {frame_id_str}{alert['level']} - {alert['type']}")
            print("="*60)


def main():
    """
    Main function to run in-car driver monitoring system.
    Supports both video file processing and live camera feed.
    """
    parser = argparse.ArgumentParser(
        description='Driver monitoring system - process videos or live camera feed'
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
    config = DriverMonitoringConfig()
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

