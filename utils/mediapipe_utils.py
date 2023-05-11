import cv2
import mediapipe as mp


def mediapipe_detection(image, model):
    """
    Use mediapipe model to extract results from OpenCV image

    param image: image gotten from OpenCV
    param model: mediapipe holistic model
    return: Return results
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_flipped = cv2.flip(image, 1)
    results = model.process(image)
    results_flipped = model.process(image_flipped)

    return results, results_flipped


def draw_landmarks(image, results):
    """
    Draw landmarks on image

    param image: image gotten from OpenCV
    param results: results from mediapipe model
    """
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    mp_drawing.draw_landmarks(
        image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(200, 250, 255), thickness=1, circle_radius=1
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 250, 150), thickness=2, circle_radius=2
        ),
    )
    mp_drawing.draw_landmarks(
        image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(200, 250, 255), thickness=1, circle_radius=2
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 250, 161), thickness=2, circle_radius=2
        ),
    )

    mp_drawing.draw_landmarks(
        image,
        landmark_list=results.pose_landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(200, 250, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 250, 150), thickness=2, circle_radius=2
        ),
    )
