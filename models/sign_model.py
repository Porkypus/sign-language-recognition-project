from typing import List

import numpy as np

from models.hand_model import HandModel
from models.pose_model import PoseModel


class SignModel(object):
    def __init__(
        self,
        left_hand_list,
        right_hand_list,
        pose_list,
    ):
        """
        Params
            x_hand_list: List of all landmarks for each frame of a video
        Args
            has_x_hand: bool; True if x hand is detected in the video, otherwise False
            xh_embedding: ndarray; Array of shape (n_frame, nb_connections * nb_connections)
        """

        self.has_left_hand = np.sum(left_hand_list) != 0
        self.has_right_hand = np.sum(right_hand_list) != 0

        self.lh_embedding = self._get_hand_embedding(left_hand_list)
        self.rh_embedding = self._get_hand_embedding(right_hand_list)
        self.left_arm_embedding = self._get_arm_embedding(pose_list)[0]
        self.right_arm_embedding = self._get_arm_embedding(pose_list)[1]

    def _get_hand_embedding(hand_list):
        """
        Params
            hand_list: List of all landmarks for each frame of a video
        Return
            Array of shape (n_frame, nb_connections * nb_connections) containing
            the feature_vectors of the hand for each frame
        """
        embedding = []
        for frame in range(len(hand_list)):
            if np.sum(hand_list[frame]) == 0:
                continue
            hand_gesture = HandModel(hand_list[frame])
            embedding.append(hand_gesture.feature_vector)
        return embedding

    def _get_arm_embedding(pose_list):
        """
        Params
            pose_list: List of all landmarks for each frame of a video
        Return
            Array of shape (n_frame, nb_connections * nb_connections) containing
            the feature_vectors of the hand for each frame
        """
        left_arm_embedding = []
        right_arm_embedding = []
        for frame in range(len(pose_list)):
            if np.sum(pose_list[frame]) == 0:
                continue
            pose = PoseModel(pose_list[frame])
            left_arm_embedding.append(pose.left_arm_embedding)
            right_arm_embedding.append(pose.right_arm_embedding)
        return left_arm_embedding, right_arm_embedding
