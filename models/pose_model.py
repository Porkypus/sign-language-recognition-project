import numpy as np


class PoseModel(object):
    def __init__(self, landmarks):

        """
        Params
            landmarks: List of all pose landmarks for a frame
        Args
            x_arm_embedding: normalized wrist landmarks of the x arm
        """

        # Reshape landmarks
        landmarks = np.array(landmarks).reshape((33, 3))

        # Selecting indices for shoulders, elbows and wrists
        # indices of left arm are 11, 13, 15
        self.left_arm_landmarks = self._normalize_landmarks(
            [landmarks[index] for index in [12, 14, 16]]
        )
        # indices of right arm are 12, 14, 16
        self.right_arm_landmarks = self._normalize_landmarks(
            [landmarks[index] for index in [11, 13, 15]]
        )

        self.left_arm_embedding = self.left_arm_landmarks[2].tolist()
        self.right_arm_embedding = self.right_arm_landmarks[2].tolist()

    def _normalize_landmarks(self, landmarks):
        """
        Normalizes dataset translation and scale.
        Indices of landmarks are as follows:
        0 - shoulder
        1 - elbow
        2 - wrist
        """
        # Take shoulder's position as origin by subtracting its coordinates from all other landmarks
        shoulder_ = landmarks[0]
        landmarks -= shoulder_

        # Divide positions by the distance between the shoulder and the elbow
        landmark_from = landmarks[0]
        landmark_to = landmarks[1]
        arm_size = np.linalg.norm(landmark_to - landmark_from)
        landmarks /= arm_size

        return landmarks
