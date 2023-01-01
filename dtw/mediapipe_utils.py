import cv2
import mediapipe as mp


def mediapipe_detection(image, model):
    """
    Use mediapipe model to get results from OpenCV image

    :param image: image gotten from OpenCV
    :param model: mediapipe holistic model
    :return: Return image and results
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Draw left hand connections
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=2, circle_radius=2
        ),
    )
    # Draw right hand connections
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=2, circle_radius=2
        ),
    )
    # Draw face connections
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.face_landmarks,
        connections=mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=1
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=1, circle_radius=1
        ),
    )
    # Draw pose connections
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.pose_landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=2, circle_radius=2
        ),
    )
    return image

    # def draw_landmarks(image, results):
    #     mp_drawing.draw_landmarks(
    #         image,
    #         results.face_landmarks,
    #         mp_holistic.FACEMESH_CONTOURS,
    #         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
    #         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
    #     )
    #     mp_drawing.draw_landmarks(
    #         image,
    #         results.left_hand_landmarks,
    #         mp_holistic.HAND_CONNECTIONS,
    #         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    #         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
    #     )
    #     mp_drawing.draw_landmarks(
    #         image,
    #         results.right_hand_landmarks,
    #         mp_holistic.HAND_CONNECTIONS,
    #         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    #         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
    #     )
    #     mp_drawing.draw_landmarks(
    #         image,
    #         results.pose_landmarks,
    #         mp_holistic.POSE_CONNECTIONS,
    #         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
    #         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
    #     )
