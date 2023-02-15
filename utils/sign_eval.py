from collections import Counter

from utils.compute_dtw import dtw_distances
from utils.compute_fastdtw import fastdtw_distances
from utils.feature_extraction import extract_landmarks
from models.sign_model import SignModel
from utils.constants import BATCH_SIZE, FASTDTW


class SignEval(object):
    """
    A class used to evaluate a set of test videos

    ...

    Attributes
    ----------
    eval_results : list
        List of results stored in each frame
    reference_signs : pd.DataFrame
        DataFrame storing the distances between the recorded sign and all the reference signs from the training dataset

    Methods
    -------
    append_eval_results(results)
        Append the results of the current frame to the list of results
    process_eval_results()
        Compute the distances between the recorded sign and all the reference signs using results, returns the predicted sign
    compute_distances()
        Updates the distance column of the reference_signs
    reset()
        Resets the variables
    get_sign_predicted(batch_size=5, threshold=0.4)
        Method that outputs the sign that appears the most in the batch_size closest reference signs
    """

    def __init__(self, reference_signs):
        """
        param reference_signs: pd.DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        """
        self.eval_results = []
        self.reference_signs = reference_signs
        self.reference_signs["distance"].values[:] = 0

    def append_eval_results(self, results):
        """
        Append the results of the current frame to the list of results

        param results: mediapipe output
        """
        self.eval_results.append(results)

    def process_eval_results(self):
        """
        Compute the distances between the recorded sign and all the reference signs using results, returns the predicted sign

        return: Return the word predicted
        """
        self.compute_distances()
        return self.get_sign_predicted()

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        """
        pose_list, left_hand_list, right_hand_list = [], [], []

        for result in self.eval_results:
            pose, left_hand, right_hand = extract_landmarks(result)
            pose_list.append(pose)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the extracted landmarks
        recorded_sign = SignModel(left_hand_list, right_hand_list, pose_list)

        # Compute sign similarity with dtw-python
        if FASTDTW:
            self.reference_signs = fastdtw_distances(
                recorded_sign, self.reference_signs
            )
        else:
            self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)

    def reset(self):
        """
        Resets the variables
        """
        self.eval_results = []
        self.reference_signs["distance"].values[:] = 0

    def get_sign_predicted(self, batch_size=BATCH_SIZE, threshold=0.4):
        """
        Outputs the sign that appears the most in the list of closest reference signs, only if its proportion within the batch is greater than the threshold

        param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
        param threshold: If the proportion of the most represented sign in the batch is greater than threshold, we output the sign_name

        return: predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()

        predicted_sign, count = sign_counter[0]
        if count / batch_size < threshold:
            return "Unknown Sign"
        return predicted_sign
