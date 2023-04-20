import os
import cv2
import pickle as pkl
import pandas as pd
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from models.sign_model import SignModel
from utils.mediapipe_utils import mediapipe_detection
from utils.constants import DATA_PATH, FEATURE_PATH


def extract_features():
    """
    Create a dataset of the videos in the training set
    """
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

    print("\nDone extracting landmarks\n")


def load_reference_signs():
    """
    Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    """
    reference_signs = {"name": [], "sign_model": [], "distance": []}
    for sign in os.listdir(FEATURE_PATH):
        for video in os.listdir(os.path.join(FEATURE_PATH, sign)):
            path = os.path.join(FEATURE_PATH, sign, video)

            left_hand_list = load_array(os.path.join(path, f"lh_{video}.pickle"))
            right_hand_list = load_array(os.path.join(path, f"rh_{video}.pickle"))
            pose_list = load_array(os.path.join(path, f"pose_{video}.pickle"))

            reference_signs["name"].append(sign)
            reference_signs["sign_model"].append(
                SignModel(left_hand_list, right_hand_list, pose_list)
            )
            reference_signs["distance"].append(0)
    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    print(
        f'Dictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}'
    )
    return reference_signs


def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)


def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    """
    Extract the results abd convert them to np arrays for each hand and the pose

    param results: mediapipe object that contains the 3D position of all keypoints
    return: 3 np arrays of size (nb_keypoints x 3)
    """

    pose = np.zeros(99).tolist()
    if results.pose_landmarks:
        pose = landmark_to_array(results.pose_landmarks).reshape(99).tolist()

    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = (
            landmark_to_array(results.right_hand_landmarks).reshape(63).tolist()
        )
    return pose, left_hand, right_hand


def save_landmarks_from_video(video_name):
    """
    Extract the landmarks from a video and save them in a pickle file
    """

    pose_list, left_hand_list, right_hand_list = [], [], []
    pose_f_list, left_hand_f_list, right_hand_f_list = [], [], []
    sign_name = video_name.split("-")[0]
    video = video_name.split(".")[0]

    # Create the folder of the sign if it doesn't exists
    path = os.path.join(FEATURE_PATH, sign_name)
    os.makedirs(path, exist_ok=True)

    # Create the folder of the video data if it doesn't exists
    data_path = os.path.join(path, video)
    data_path_flipped = os.path.join(path, f"{video}-flipped")

    # Check if the video data already exists
    if not os.path.exists(data_path) and not os.path.exists(data_path_flipped):
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(data_path_flipped, exist_ok=True)
        # Set the Video stream
        cap = cv2.VideoCapture(os.path.join(DATA_PATH, sign_name, video_name))
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Make detections
                    results, results_flipped = mediapipe_detection(frame, holistic)

                    # Store results
                    pose, left_hand, right_hand = extract_landmarks(results)
                    pose_list.append(pose)
                    left_hand_list.append(left_hand)
                    right_hand_list.append(right_hand)

                    pose_f, left_hand_f, right_hand_f = extract_landmarks(
                        results_flipped
                    )
                    pose_f_list.append(pose_f)
                    left_hand_f_list.append(left_hand_f)
                    right_hand_f_list.append(right_hand_f)

                else:
                    break
            cap.release()

        # Save landmarks into pickle files
        save_array(pose_list, os.path.join(data_path, f"pose_{video}.pickle"))
        save_array(
            left_hand_list,
            os.path.join(data_path, f"lh_{video}.pickle"),
        )
        save_array(
            right_hand_list,
            os.path.join(data_path, f"rh_{video}.pickle"),
        )
        save_array(
            pose_f_list, os.path.join(data_path_flipped, f"pose_{video}-flipped.pickle")
        )
        save_array(
            left_hand_f_list,
            os.path.join(data_path_flipped, f"lh_{video}-flipped.pickle"),
        )
        save_array(
            right_hand_f_list,
            os.path.join(data_path_flipped, f"rh_{video}-flipped.pickle"),
        )
