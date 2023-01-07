import pandas as pd
import numpy as np
from collections import Counter

from utils.compute_dtw import dtw_distances
from models.sign_model import SignModel
from utils.mediapipe_utils import extract_landmarks


class SignEval(object):
    def __init__(self, reference_signs: pd.DataFrame):

        # List of results stored each frame
        self.eval_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs
        self.reference_signs["distance"].values[:] = 0

    def append_eval_results(self, results):
        self.eval_results.append(results)

    def process_eval_results(self):
        self.compute_eval_distances()
        print(self.reference_signs)

        return self._get_sign_predicted()

    def compute_eval_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        pose_list, left_hand_list, right_hand_list = [], [], []
        for results in self.eval_results:
            pose, left_hand, right_hand = extract_landmarks(results)
            pose_list.append(pose)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(left_hand_list, right_hand_list, pose_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)

    def reset(self):
        self.eval_results = []
        self.reference_signs["distance"].values[:] = 0

    def _get_sign_predicted(self, batch_size=5, threshold=0.4):
        """
        Method that outputs the sign that appears the most in the list of closest
        reference signs, only if its proportion within the batch is greater than the threshold

        :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
        :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                        we output the sign_name
                          If not,
                        we output "Sign not found"
        :return: The name of the predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()

        predicted_sign, count = sign_counter[0]
        if count / batch_size < threshold:
            return "Unknown Sign"
        return predicted_sign
