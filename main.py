import cv2
import mediapipe
from utils.feature_extraction import extract_features, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from utils.sign_recorder import SignRecorder
from utils.webcam_manager import WebcamManager

# Create dataset of the videos where landmarks have not been extracted yet
videos = extract_features()

# Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
reference_signs = load_reference_signs()

# Object that stores mediapipe results and computes sign similarities
sign_recorder = SignRecorder(reference_signs)

# Object that draws keypoints & displays results
webcam_manager = WebcamManager()

# Turn on the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set up the Mediapipe environment
with mediapipe.solutions.holistic.Holistic(
    min_detection_confidence=0.7, min_tracking_confidence=0.7
) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        results, _ = mediapipe_detection(frame, holistic)

        # Process results
        sign_detected, is_recording = sign_recorder.process_results(results)

        # Update the frame (draw landmarks & display result)
        webcam_manager.update(frame, results, sign_detected, is_recording)

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord("r"):  # Record pressing r
            sign_recorder.record()
        elif pressedKey == ord("q"):  # Break pressing q
            break

    cap.release()
    cv2.destroyAllWindows()
