import cv2
import mediapipe
import os
from tqdm import tqdm
from utils.feature_extraction import extract_features, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from utils.sign_eval import SignEval
from utils.constants import TEST_PATH

# Create dataset of the videos where landmarks have not been extracted yet
total = 0
correct = 0
sign_correct = {}

videos = extract_features()

# Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
reference_signs = load_reference_signs()

# Object that stores mediapipe results and computes sign similarities
sign_eval = SignEval(reference_signs)

for sign in tqdm(os.listdir(TEST_PATH), desc="Evaluating Signs", position=2):
    for video in os.listdir(os.path.join(TEST_PATH, sign)):
        total += 1
        cap = cv2.VideoCapture(os.path.join(TEST_PATH, sign, video))
        with mediapipe.solutions.holistic.Holistic(
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        ) as holistic:
            count = 0
            while cap.isOpened():
                # Read feed
                count += 1
                ret, frame = cap.read()
                # frame = cv2.flip(frame, 1)
                if not ret:
                    break
                # Make detections
                results, _ = mediapipe_detection(frame, holistic)

                # Process results
                sign_eval.append_eval_results(results)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        sign_detected = sign_eval.process_eval_results()
        sign_eval.reset()
        if sign_detected == sign:
            sign_correct[sign] = sign_correct.get(sign, 0) + 1
            correct += 1

        cap.release()
        cv2.destroyAllWindows()

print(f"Accuracy: {correct/total}")
print(sign_correct)
