import os
import shutil
from constants import mappings

os.makedirs("train_videos", exist_ok=True)
os.makedirs("test_videos", exist_ok=True)
count = 0
for video in os.listdir("datasets/lsa64_dataset"):
    word = mappings[video[:3]]
    if count < 40:
        os.makedirs(os.path.join("train_videos", word), exist_ok=True)
        shutil.copy(
            os.path.join("datasets/lsa64_dataset", video),
            os.path.join(
                "train_videos",
                word,
                word + "-" + str(count) + ".mp4",
            ),
        )
        count += 1
    elif count < 49:
        os.makedirs(os.path.join("test_videos", word), exist_ok=True)
        shutil.copy(
            os.path.join("datasets/lsa64_dataset", video),
            os.path.join(
                "test_videos",
                word,
                word + "-" + str(count) + ".mp4",
            ),
        )
        count += 1
    else:
        os.makedirs(os.path.join("test_videos", word), exist_ok=True)
        shutil.copy(
            os.path.join("datasets/lsa64_dataset", video),
            os.path.join(
                "test_videos",
                word,
                word + "-" + str(count) + ".mp4",
            ),
        )
        count = 0
