import os

import pandas as pd
from tqdm import tqdm
from models.sign_model import SignModel
from utils.mediapipe_utils import save_landmarks_from_video, load_array
from utils.constants import DATA_PATH, FEATURE_PATH


def load_dataset():
    videos = [
        video
        for sign in os.listdir(DATA_PATH)
        for video in os.listdir(os.path.join(DATA_PATH, sign))
    ]

    # Create the dataset from the reference videos
    n = len(videos)
    if n > 0:
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")
        for i in tqdm(range(n)):
            save_landmarks_from_video(videos[i])

    return videos


def load_reference_signs(videos):
    reference_signs = {"name": [], "sign_model": [], "distance": []}
    for video_name in videos:
        sign_name = video_name.split("-")[0]
        path = os.path.join(FEATURE_PATH, sign_name, video_name)

        left_hand_list = load_array(os.path.join(path, f"lh_{video_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_name}.pickle"))
        # pose_list = load_array(os.path.join(path, f"pose_{video_name}.pickle"))

        reference_signs["name"].append(sign_name)
        reference_signs["sign_model"].append(SignModel(left_hand_list, right_hand_list))
        reference_signs["distance"].append(0)

    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    print(
        f'Dictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}'
    )
    return reference_signs
