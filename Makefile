
run_default:
	python examples/in-car-driver-monitoring.py

# Process live camera feed:
run_camera: 
	python examples/in-car-driver-monitoring.py --camera

# Process camera feed with a different camera device:
run_camera_index:
	python examples/in-car-driver-monitoring.py --camera --camera-index 1

# Process camera feed without display (faster):
process_no_display:
	python examples/in-car-driver-monitoring.py --camera --no-display

# Process a specific video file:
process_video_name:
	python examples/in-car-driver-monitoring-fast.py --video data/30-11-2025/001.mp4

