from dtw import dtw
import numpy as np


def dtw_distances(recorded_sign, reference_signs):
    """
    Use fastdtw to compute similarity between the recorded sign and the reference signs

    param recorded_sign: SignModel object containing extracted data from the recorded sign
    param reference_signs: pd.DataFrame (name: str, model: SignModel, distance: int)

    return: Return a sign dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding

    rec_left_arm = recorded_sign.left_arm_embedding
    rec_right_arm = recorded_sign.right_arm_embedding

    for _, row in reference_signs.iterrows():
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

            # dtw-python transposes the list of feature vectors if the number of feature vectors is 1.
            # This is inherent to the data collection, determined by the confidence tracking of mediapipe
            if recorded_sign.has_left_hand:
                if len(rec_left_hand) == 1:
                    rec_left_hand.append(rec_left_hand[0])
                if len(ref_left_hand) == 1:
                    ref_left_hand.append(ref_left_hand[0])
                row["distance"] += dtw(rec_left_hand, ref_left_hand).distance
            if recorded_sign.has_right_hand:
                if len(rec_right_hand) == 1:
                    rec_right_hand.append(rec_right_hand[0])
                if len(ref_right_hand) == 1:
                    ref_right_hand.append(ref_right_hand[0])
                row["distance"] += dtw(rec_right_hand, ref_right_hand).distance

            row["distance"] += dtw(rec_left_arm, ref_left_arm).distance
            row["distance"] += dtw(rec_right_arm, ref_right_arm).distance

        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])
