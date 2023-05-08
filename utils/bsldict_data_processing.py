import os
import shutil
from constants import mappings

os.makedirs("train_videos", exist_ok=True)
os.makedirs("test_videos", exist_ok=True)

# Number of training videos per word
n = 7

path = os.path.join("datasets", "bsl_processed_dataset")
for sign in os.listdir(path):
    count = 0
    for video in os.listdir(os.path.join(path, sign)):
        word = video.split("-")[0]
        if count < n:
            os.makedirs(os.path.join("train_videos", word), exist_ok=True)
            shutil.copy(
                os.path.join(path, sign, video),
                os.path.join(
                    "train_videos",
                    word,
                    word + "-" + str(count) + ".mp4",
                ),
            )

        else:
            os.makedirs(os.path.join("test_videos", word), exist_ok=True)
            shutil.copy(
                os.path.join(path, sign, video),
                os.path.join(
                    "test_videos",
                    word,
                    word + "-" + str(count) + ".mp4",
                ),
            )
        count += 1
