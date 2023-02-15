from fastdtw import fastdtw
import numpy as np


def fastdtw_distances(recorded_sign, reference_signs):
    """
    Use dtw-python to compute similarity between the recorded sign and the reference signs

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
        _, ref_sign_model, _ = row

        # check if the reference sign has the same number of hands
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

        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])
