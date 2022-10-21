import cv2
from constants import mp_holistic
from visulisation import mediapipe_detection, draw_landmarks

# Select a webcam
cap = cv2.VideoCapture(0)

# Access mediapipe model
with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    smooth_landmarks=True,
) as holistic:
    while cap.isOpened():
        # Read feed
        # ret is return value, frame is the image
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        cv2.imshow("OpenCV Feed", image)

        # Break using q
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
