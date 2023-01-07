import os
import cv2
from utils.mediapipe_utils import mediapipe_detection
from utils.feature_extraction import load_dataset, load_reference_signs
from utils.sign_eval import SignEval
from utils.mediapipe_utils import draw_landmarks
import mediapipe


videos = load_dataset()

# Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
reference_signs = load_reference_signs(videos)

# Object that stores mediapipe results and computes sign similarities
sign_recorder = SignEval(reference_signs)

cap = cv2.VideoCapture(os.path.join("test_eval", "Accept", "Accept-2.mp4"))
with mediapipe.solutions.holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            break
        # Make detections
        # frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        cv2.imshow("Video", image)
        # Process results
        sign_recorder.append_eval_results(results)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

sign_detected = sign_recorder.process_eval_results()
sign_recorder.reset()
print(sign_detected)

cap.release()
cv2.destroyAllWindows()
