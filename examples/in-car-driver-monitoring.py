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
    
    def detect_impaired_driving(self, frames: List[bytes], face_state: Optional[str] = None, frame_id: Optional[int] = None) -> Dict[str, any]:
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
            "Analyze these sequential video frames for signs of impaired driving that pose a safety risk. "
            "Look for: 1) Eyes closed or heavily drooping 2) Head nodding or falling forward "
            "3) Erratic head movements 4) Slowed reactions or unresponsiveness "
            "5) Signs of drowsiness, exhaustion, fatigue, or emotional distress affecting driving ability. "
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
            "Rate the concern level as: LOW (normal/alert driver with no safety concerns), "
            "MODERATE (mild drowsiness, tiredness, exhaustion, or distraction that could affect driving), "
            "or HIGH (severe drowsiness, eyes closed, exhaustion, or dangerous behavior). "
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
        
        result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'analysis_type': 'impaired_driving',
            'response': response,
            'alert_level': alert_level,
            'face_state_considered': face_state,
            'frames_analyzed': len(frames)
        }
        
        # Include frame_id if provided
        if frame_id is not None:
            result['frame_id'] = frame_id
        
        # Trigger alert for any concerning state
        if alert_level in ['MODERATE', 'HIGH']:
            self._trigger_alert(result)
        
        return result
    
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
        
        # Check for moderate concern keywords
        moderate_keywords = [
            'MILD DROWSINESS', 'SOMEWHAT TIRED', 'SLIGHTLY DISTRACTED',
            'TIRED', 'FATIGUE', 'EXHAUSTION', 'DISTRACTED', 'ANGRY'
        ]
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
        Sub-optimal states (tired, exhausted, sleepy, angry) should trigger alerts.
        
        Args:
            current_level: Current alert level from LLM response
            face_state: Detected face state
            response: LLM response text (may be None)
        
        Returns:
            Adjusted alert level
        """
        # Define concerning face states that should trigger alerts
        concerning_states = {
            'SLEEPY': 'HIGH',      # Sleepy is high concern
            'EXHAUSTED': 'HIGH',   # Exhausted is high concern
            'TIRED': 'MODERATE',   # Tired is moderate concern
            'ANGRY': 'MODERATE',   # Angry is moderate concern (emotional state affecting driving)
        }
        
        # Normal states that should not raise alerts unless LLM detects issues
        normal_states = ['AWAKE', 'NEUTRAL', 'JOYFUL']
        
        # If face state is concerning, adjust alert level
        if face_state in concerning_states:
            required_level = concerning_states[face_state]
            
            # If current level is lower than required, upgrade it
            level_priority = {'LOW': 1, 'MODERATE': 2, 'HIGH': 3}
            if level_priority.get(current_level, 0) < level_priority.get(required_level, 0):
                logger.info(
                    f"Upgrading alert level from {current_level} to {required_level} "
                    f"based on face state: {face_state}"
                )
                return required_level
        
        # If face state is normal but LLM detected issues, trust the LLM
        elif face_state in normal_states:
            # Keep the LLM's assessment, but don't downgrade if it's already MODERATE/HIGH
            return current_level
        
        # For unknown states, trust the LLM but be cautious
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
        
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(
            f"SAFETY ALERT - Level: {alert['level']}, "
            f"Type: {alert['type']}, "
            f"Time: {alert['timestamp']}"
        )
        
        # Print alert (in real system, this would trigger audio/visual warnings)
        print("\n" + "="*60)
        print(f"⚠️  SAFETY ALERT - {alert['level']} CONCERN DETECTED")
        print(f"Type: {alert['type']}")
        print(f"Time: {alert['timestamp']}")
        print(f"Details: {alert['details'][:200]}...")
        print("="*60 + "\n")
    
    def process_frame_sequence(self, frames: List[bytes], frame_id: Optional[int] = None) -> Dict[str, any]:
        """
        Process a sequence of frames for comprehensive driver analysis.
        
        Args:
            frames: List of frame bytes (temporal sequence)
            frame_id: Optional frame sequence ID for tracking
        
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
        # Also pass frame_id if available
        impaired_result = self.detect_impaired_driving(
            frames, 
            face_state=detected_face_state,
            frame_id=frame_id
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
        self.frame_summaries = []  # Store summaries for each analyzed frame sequence
    
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
                cv2.imshow('Driver Monitoring', frame)
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
        logger.info(f"Processing interval: {self.config.PROCESSING_INTERVAL_SECONDS} seconds")
        logger.info(f"Frame buffer size: {self.config.FRAME_BUFFER_SIZE} frames")
        
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
                
                # Perform comprehensive analysis (pass frame_id so alerts can include it)
                results = self.analyzer.process_frame_sequence(frame_sequence, frame_id=self.frame_sequence_id)
                
                # Add video frame number to results
                results['video_frame_number'] = frame_number
                
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
        
        print("\n" + "-"*60)
        print(f"Frame #{frame_id} (Video Frame {video_frame}) - {results['timestamp']}")
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


def main():
    """
    Main function to run in-car driver monitoring system.
    """
    # Initialize configuration
    config = DriverMonitoringConfig()
    
    # Initialize video processor
    processor = InCarVideoProcessor(config)
    
    # Open video source (0 for webcam, or path to video file)
    video_source = 0  # Change to video file path if using recorded video
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {video_source}")
        raise IOError(f"Cannot open video source: {video_source}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        logger.info(f"Video FPS: {fps}")
        # Update config if video has different frame rate
        if abs(fps - config.FRAME_RATE) > 5:
            logger.warning(
                f"Video FPS ({fps}) differs significantly from config ({config.FRAME_RATE}). "
                "Consider updating config."
            )
    
    try:
        # Process video stream
        processor.process_video_stream(cap, debug_show=True)
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


if __name__ == "__main__":
    main()

