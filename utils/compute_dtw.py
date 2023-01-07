import pandas as pd
from dtw import dtw
from fastdtw import fastdtw
import numpy as np
from models.sign_model import SignModel


def dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame):
    """
    Use DTW to compute similarity between the recorded sign & the reference signs

    :param recorded_sign: a SignModel object containing the data gathered during record
    :param reference_signs: pd.DataFrame
                            columns : name, dtype: str
                                      sign_model, dtype: SignModel
                                      distance, dtype: float64
    :return: Return a sign dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding

    rec_left_arm = recorded_sign.left_arm_embedding
    rec_right_arm = recorded_sign.right_arm_embedding

    for idx, row in reference_signs.iterrows():
        # Initialize the row variables
        _, ref_sign_model, _ = row

        # If the reference sign has the same number of hands compute dtw
        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ):
            ref_left_hand = ref_sign_model.lh_embedding
            ref_right_hand = ref_sign_model.rh_embedding

            ref_left_arm = ref_sign_model.left_arm_embedding
            ref_right_arm = ref_sign_model.right_arm_embedding

            if recorded_sign.has_left_hand:
                row["distance"] += list(fastdtw(rec_left_hand, ref_left_hand))[0]
            if recorded_sign.has_right_hand:
                row["distance"] += list(fastdtw(rec_right_hand, ref_right_hand))[0]

            row["distance"] += list(fastdtw(rec_left_arm, ref_left_arm))[0]
            row["distance"] += list(fastdtw(rec_right_arm, ref_right_arm))[0]

            # if recorded_sign.has_left_hand:
            #     row["distance"] += dtw(rec_left_hand, ref_left_hand).distance
            # if recorded_sign.has_right_hand:
            #     row["distance"] += dtw(rec_right_hand, ref_right_hand).distance

            # row["distance"] += dtw(rec_left_arm, ref_left_arm).distance
            # row["distance"] += dtw(rec_right_arm, ref_right_arm).distance

        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])
