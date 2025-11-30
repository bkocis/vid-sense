"""
Video Recording Utility

This utility provides functions to record video feeds from webcams or video files
and save them as MP4 files in the data folder for testing purposes.

Usage:
    # Record from webcam for 10 seconds
    record_video_from_source(0, duration_seconds=10, output_filename="test_recording.mp4")
    
    # Record from video file (re-encode)
    record_video_from_source("input_video.mp4", output_filename="test_recording.mp4")
    
    # Record from webcam until user stops (press 'q')
    record_video_from_source(0, duration_seconds=None, output_filename="test_recording.mp4")
"""

import cv2
import os
import datetime
import time
from pathlib import Path
from typing import Union, Optional, Tuple


def ensure_data_folder() -> Path:
    """
    Ensure the data folder exists, create it if it doesn't.
    
    Returns:
        Path object pointing to the data folder
    """
    data_folder = Path(__file__).parent / "data"
    data_folder.mkdir(exist_ok=True)
    return data_folder


def measure_actual_fps(cap: cv2.VideoCapture, sample_frames: int = 30) -> Tuple[float, list]:
    """
    Measure the actual frame capture rate by reading and timing frames.
    This gives us the real FPS that the source is providing.
    Also returns the frames read so they can be used in the recording.
    
    Args:
        cap: OpenCV VideoCapture object
        sample_frames: Number of frames to sample for measurement
        
    Returns:
        Tuple of (measured_fps, list_of_frames_read)
    """
    print(f"Measuring actual capture FPS (sampling {sample_frames} frames)...")
    
    frame_times = []
    frames_read = []
    
    for i in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        frame_times.append(current_time)
        frames_read.append(frame.copy())  # Store frame for later use
    
    if len(frame_times) < 5:
        print("Warning: Could not measure FPS (not enough frames), using default 30.0")
        return 30.0, frames_read
    
    # Calculate FPS from frame intervals
    intervals = []
    for i in range(1, len(frame_times)):
        interval = frame_times[i] - frame_times[i-1]
        if interval > 0:
            intervals.append(interval)
    
    if not intervals:
        print("Warning: Could not calculate FPS intervals, using default 30.0")
        return 30.0, frames_read
    
    # Use median interval to avoid outliers
    intervals.sort()
    median_interval = intervals[len(intervals) // 2]
    measured_fps = 1.0 / median_interval if median_interval > 0 else 30.0
    
    # Validate measured FPS is reasonable
    if measured_fps < 5 or measured_fps > 120:
        print(f"Warning: Measured FPS ({measured_fps:.2f}) seems unrealistic, using default 30.0")
        return 30.0, frames_read
    
    print(f"Measured capture FPS: {measured_fps:.2f}")
    return measured_fps, frames_read


def get_video_properties(cap: cv2.VideoCapture, is_webcam: bool = False, measure_fps: bool = True) -> Tuple[int, int, float, list]:
    """
    Get video properties from a VideoCapture object.
    For webcams, trusts reported FPS if reasonable, otherwise measures actual FPS.
    For video files, reads from metadata.
    
    Args:
        cap: OpenCV VideoCapture object
        is_webcam: Whether this is a webcam (True) or video file (False)
        measure_fps: Whether to measure actual FPS if reported FPS is unreliable (default: True)
        
    Returns:
        Tuple of (width, height, fps, calibration_frames)
        calibration_frames: Frames read during FPS measurement (empty list if not measured)
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    calibration_frames = []
    
    # For webcams, check if reported FPS is reasonable
    if is_webcam:
        # If FPS is reported and reasonable (between 10 and 60), trust it
        if 10 <= fps <= 60:
            print(f"Using webcam reported FPS: {fps:.2f}")
            # No need to measure, just use the reported value
        elif measure_fps:
            # FPS is unreliable (0, negative, or unrealistic), measure it
            print(f"Webcam reported FPS ({fps:.2f}) is unreliable, measuring actual FPS...")
            measured_fps, calibration_frames = measure_actual_fps(cap, sample_frames=15)
            fps = measured_fps
        else:
            # Measurement disabled, use default
            fps = 30.0
            print(f"Using default FPS: {fps:.2f}")
    else:
        # For video files, use the FPS from the file
        # Ensure it's reasonable
        if fps <= 0 or fps > 120:
            fps = 30.0
            print(f"Video file FPS was invalid, using default: {fps:.2f}")
        else:
            print(f"Using video file FPS: {fps:.2f}")
    
    return width, height, fps, calibration_frames


def get_best_codec() -> str:
    """
    Try to find the best available codec for MP4 recording.
    H.264 is preferred for best compatibility and playback performance.
    
    Returns:
        FourCC codec string
    """
    # Try codecs in order of preference
    codecs_to_try = [
        ('avc1', 'H.264 (avc1) - Best compatibility'),
        ('H264', 'H.264 (H264) - Alternative H.264'),
        ('mp4v', 'MPEG-4 (mp4v) - Fallback'),
        ('XVID', 'XVID - Last resort'),
    ]
    
    # Test codec by trying to create a VideoWriter
    test_path = str(Path(__file__).parent / "data" / ".codec_test.mp4")
    ensure_data_folder()
    
    for fourcc_str, description in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            # Try to create a test writer
            test_writer = cv2.VideoWriter(test_path, fourcc, 30.0, (640, 480))
            if test_writer.isOpened():
                test_writer.release()
                # Clean up test file
                try:
                    os.remove(test_path)
                except:
                    pass
                print(f"Selected codec: {description}")
                return fourcc_str
            test_writer.release()
        except Exception as e:
            continue
    
    # Default fallback
    print("Warning: Using default codec 'mp4v' (may have playback issues)")
    return 'mp4v'


def record_video_from_source(
    video_source: Union[int, str],
    output_filename: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    show_preview: bool = True,
    data_folder: Optional[Union[str, Path]] = None
) -> str:
    """
    Record video from a webcam or video file and save it as MP4.
    
    Args:
        video_source: Video source (0 for webcam, or path to video file)
        output_filename: Name of the output file (default: timestamp-based name)
        duration_seconds: Duration to record in seconds (None = record until stopped)
        show_preview: Whether to show a preview window while recording
        data_folder: Custom data folder path (default: ./data)
        
    Returns:
        Path to the saved video file
        
    Raises:
        IOError: If video source cannot be opened
        ValueError: If duration is invalid
    """
    # Ensure data folder exists
    if data_folder is None:
        data_folder = ensure_data_folder()
    else:
        data_folder = Path(data_folder)
        data_folder.mkdir(exist_ok=True)
    
    # Generate output filename if not provided
    if output_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"recording_{timestamp}.mp4"
    
    # Ensure filename has .mp4 extension
    if not output_filename.endswith('.mp4'):
        output_filename += '.mp4'
    
    output_path = data_folder / output_filename
    
    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_source}")
    
    # Determine if this is a webcam (integer) or video file (string path)
    is_webcam = isinstance(video_source, int)
    
    # Get video properties (measure actual FPS for webcams to ensure correct playback speed)
    width, height, fps, calibration_frames = get_video_properties(cap, is_webcam=is_webcam, measure_fps=True)
    
    print(f"Recording from source: {video_source}")
    print(f"Video properties: {width}x{height} @ {fps:.2f} FPS")
    print(f"Output file: {output_path}")
    if duration_seconds:
        print(f"Duration: {duration_seconds} seconds")
    else:
        print("Duration: Until stopped (press 'q' to stop)")
    
    # Get best available codec (H.264 preferred)
    codec_str = get_best_codec()
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    
    # Create VideoWriter with proper settings (using measured FPS)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise IOError(
            f"Failed to create video writer for: {output_path}\n"
            f"This may be due to codec '{codec_str}' not being available. "
            f"Try installing codec support or using a different format."
        )
    
    frame_count = 0
    start_time = time.time()
    last_frame_time = start_time
    frame_interval = 1.0 / fps  # Time between frames for proper timing
    
    # If we used calibration frames, we measured FPS and should enforce timing
    # If we trusted reported FPS, the natural capture rate should match, so less strict control
    use_strict_timing = len(calibration_frames) > 0
    
    print(f"Recording started... (Target FPS: {fps:.2f}, Frame interval: {frame_interval:.4f}s)")
    if use_strict_timing:
        print("Using measured FPS with strict timing control")
    else:
        print("Using reported FPS - writing at natural capture rate")
    
    try:
        # Write calibration frames first (frames used for FPS measurement)
        for cal_frame in calibration_frames:
            out.write(cal_frame)
            frame_count += 1
        
        # Continue recording from where we left off
        # Update last_frame_time to account for calibration frames
        if calibration_frames:
            last_frame_time = time.time()
        
        while True:
            # Only enforce strict timing if we measured FPS
            # Otherwise, trust the natural capture rate
            if use_strict_timing:
                # Calculate when the next frame should be written
                next_frame_time = last_frame_time + frame_interval
                current_time = time.time()
                
                # Wait if we're ahead of schedule (to maintain proper frame timing)
                if current_time < next_frame_time:
                    sleep_time = next_frame_time - current_time
                    if sleep_time > 0.001:  # Only sleep if significant (>1ms)
                        time.sleep(sleep_time)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame or end of video reached")
                break
            
            # Write frame to output video
            out.write(frame)
            frame_count += 1
            
            # Update timing for next frame
            last_frame_time = time.time()
            
            # Show preview if requested
            if show_preview:
                # Add recording indicator
                cv2.putText(
                    frame, 
                    "RECORDING", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
                cv2.imshow('Recording Preview (Press q to stop)', frame)
            
            # Check for duration limit
            if duration_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    print(f"\nRecording duration ({duration_seconds}s) reached")
                    break
            
            # Check for user stop (q key)
            if show_preview and cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRecording stopped by user")
                break
        
        # Calculate actual statistics
        total_time = time.time() - start_time
        actual_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\nRecording complete!")
        print(f"  Frames saved: {frame_count}")
        print(f"  Duration: {total_time:.2f} seconds")
        print(f"  Actual FPS: {actual_fps:.2f}")
        print(f"  Video saved to: {output_path}")
        
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    finally:
        # Cleanup - ensure all frames are written
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Verify file was created and has content
        if output_path.exists():
            file_size = output_path.stat().st_size
            if file_size > 0:
                print(f"  File size: {file_size / (1024*1024):.2f} MB")
            else:
                print("  WARNING: Output file is empty!")
        else:
            print("  ERROR: Output file was not created!")
    
    return str(output_path)


def record_from_webcam(
    duration_seconds: Optional[float] = None,
    output_filename: Optional[str] = None,
    camera_index: int = 0,
    show_preview: bool = True
) -> str:
    """
    Convenience function to record from webcam.
    
    Args:
        duration_seconds: Duration to record in seconds (None = record until stopped)
        output_filename: Name of the output file (default: timestamp-based name)
        camera_index: Camera index (default: 0)
        show_preview: Whether to show a preview window while recording
        
    Returns:
        Path to the saved video file
    """
    return record_video_from_source(
        video_source=camera_index,
        output_filename=output_filename,
        duration_seconds=duration_seconds,
        show_preview=show_preview
    )


def record_from_video_file(
    input_video_path: str,
    output_filename: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    show_preview: bool = True
) -> str:
    """
    Convenience function to re-encode a video file (useful for format conversion).
    
    Args:
        input_video_path: Path to input video file
        output_filename: Name of the output file (default: timestamp-based name)
        duration_seconds: Duration to record in seconds (None = record entire video)
        show_preview: Whether to show a preview window while recording
        
    Returns:
        Path to the saved video file
    """
    return record_video_from_source(
        video_source=input_video_path,
        output_filename=output_filename,
        duration_seconds=duration_seconds,
        show_preview=show_preview
    )


if __name__ == "__main__":
    """
    Example usage of the recording utility.
    """
    import sys
    
    print("Video Recording Utility")
    print("=" * 50)
    
    # Example 1: Record from webcam for 10 seconds
    if len(sys.argv) > 1 and sys.argv[1] == "webcam":
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        print(f"\nRecording from webcam for {duration} seconds...")
        output = record_from_webcam(duration_seconds=duration)
        print(f"Saved to: {output}")
    
    # Example 2: Record from video file
    elif len(sys.argv) > 1 and sys.argv[1] == "file":
        if len(sys.argv) < 3:
            print("Usage: python recording_utility.py file <input_video_path> [duration_seconds]")
            sys.exit(1)
        input_path = sys.argv[2]
        duration = float(sys.argv[3]) if len(sys.argv) > 3 else None
        print(f"\nRe-encoding video file: {input_path}")
        output = record_from_video_file(input_path, duration_seconds=duration)
        print(f"Saved to: {output}")
    
    # Default: Interactive recording (until 'q' is pressed)
    else:
        print("\nRecording from webcam (press 'q' to stop)...")
        output = record_from_webcam(duration_seconds=None)
        print(f"Saved to: {output}")

