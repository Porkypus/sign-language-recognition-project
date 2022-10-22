import cv2
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import os
from constants import DATA_PATH, DATASET_PATH, sequence_length, mp_holistic
from asl_data_parsing import get_videos_ids, get_json_features
from visualisation_methods import mediapipe_detection, draw_landmarks, extract_keypoints


wlas_df = pd.read_json(DATASET_PATH + "WLASL_v0.3.json")
wlas_df["videos_ids"] = wlas_df["instances"].apply(get_videos_ids)

features_df = pd.DataFrame(columns=["gloss", "video_id", "url"])
for row in wlas_df.iterrows():
    ids, urls = get_json_features(row[1][1])
    word = [row[1][0]] * len(ids)
    df = pd.DataFrame(list(zip(word, ids, urls)), columns=features_df.columns)
    # features_df = features_df.append(df, ignore_index=True)
    features_df = pd.concat([features_df, df], ignore_index=True)
    features_df.index.name = "index"

for row in features_df.iterrows():
    name = row[1][0]
    video_id = row[1][1]
    path = DATASET_PATH + "videos/" + video_id + ".mp4"

    cap = cv2.VideoCapture(path)

    # Access mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        smooth_landmarks=True,
    ) as holistic:
        for frame_num in range(sequence_length):
            # Read feed
            # ret is return value, frame is the image
            ret, frame = cap.read()
            if not ret:
                break
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, name, video_id, str(frame_num))
            os.makedirs(os.path.join(DATA_PATH, name, video_id), exist_ok=True)
            np.save(npy_path, keypoints)

            # Break using q
            if cv2.waitKey(10) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                break

print("Data created successfully!")
