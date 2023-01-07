import cv2
import mediapipe
import os

from utils.feature_extraction import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from utils.sign_eval import SignEval
from utils.mediapipe_utils import draw_landmarks


if __name__ == "__main__":
    # Create dataset of the videos where landmarks have not been extracted yet
    total = 0
    correct = 0

    videos = load_dataset()

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    reference_signs = load_reference_signs(videos)

    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignEval(reference_signs)

    for sign in os.listdir("test_eval"):
        for video in os.listdir(os.path.join("test_eval", sign)):
            total += 1
            cap = cv2.VideoCapture(os.path.join("test_eval", sign, video))
            with mediapipe.solutions.holistic.Holistic(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as holistic:
                count = 0
                while cap.isOpened():
                    # Read feed
                    count += 1
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)
                    cv2.imshow("Video", image)
                    # Process results
                    sign_recorder.append_eval_results(results)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            sign_detected = sign_recorder.process_eval_results()
            sign_recorder.reset()
            if sign_detected == sign:
                correct += 1

            cap.release()
            cv2.destroyAllWindows()

    print(f"Accuracy: {correct/total}")
