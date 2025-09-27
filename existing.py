from drone_classifier import DroneDetector

# Create detector once
detector = DroneDetector()

# Use it in your existing code
if detector.detect_drone("test_image.jpg"):
    print("Drone found!")
    # Your existing code here