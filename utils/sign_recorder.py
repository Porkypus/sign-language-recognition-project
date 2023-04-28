import numpy as np
from collections import Counter
from utils.compute_dtw import dtw_distances
from utils.compute_fastdtw import fastdtw_distances
from utils.feature_extraction import extract_landmarks
from models.sign_model import SignModel
from utils.constants import BATCH_SIZE, THRESHOLD, FASTDTW


class SignRecorder(object):
    """
    A class used to predict a recorded sign

    ...

    Attributes
    ----------
    is_recording : bool
        Boolean that indicates if the SignRecorder is recording a sign
    seq_len : int
        Number of frames to record
    recorded_results : list
        List of results stored in each frame
    reference_signs : pd.DataFrame
        DataFrame storing the distances between the recorded sign and all the reference signs from the training dataset

    Methods
    -------
    record()
        Initialize sign_distances & start recording
    process_results()
        Compute the distances between the recorded sign and all the reference signs using results, returns the predicted sign
    compute_distances()
        Updates the distance column of the reference_signs
    get_sign_predicted(batch_size=5, threshold=0.4)
        Method that outputs the sign that appears the most in the batch_size closest reference signs
    """

    def __init__(self, reference_signs, seq_len=50):
        """
        param reference_signs: pd.DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        param seq_len: Number of frames to record
        """
        self.is_recording = False
        self.seq_len = seq_len
        self.recorded_results = []
        self.reference_signs = reference_signs

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results):
        """
        If recording, append the results of the current frame to the list of results, else
        compute the distances between the recorded sign and all the reference signs using results, returns the predicted sign

        param results: mediapipe output
        return: Return the word predicted (blank text if there is no distances) and the recording status
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.compute_distances()
                print(self.reference_signs.iloc[0:10])

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return self.get_sign_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs and resets recording variables
        """
        pose_list, left_hand_list, right_hand_list = [], [], []

        for result in self.recorded_results:
            pose, left_hand, right_hand = extract_landmarks(result)
            pose_list.append(pose)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(left_hand_list, right_hand_list, pose_list)

        # Compute sign similarity with DTW (ascending order)
        if FASTDTW:
            self.reference_signs = fastdtw_distances(
                recorded_sign, self.reference_signs
            )
        else:
            self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)

        # Reset variables
        self.recorded_results = []
        self.is_recording = False

    def get_sign_predicted(self, batch_size=BATCH_SIZE, threshold=THRESHOLD):
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
