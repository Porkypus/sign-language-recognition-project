import os
import shutil
from constants import mappings

os.makedirs(os.path.join("datasets", "bsl_processed_dataset"), exist_ok=True)

n = 5
count = {}
downloaded = {}

for video in os.listdir("datasets/bsl_dataset"):
    word = video.split("_")[4].split(".")[0]
    if word not in count:
        count[word] = 0
    count[word] += 1

for video in os.listdir("datasets/bsl_dataset"):
    word = video.split("_")[4].split(".")[0]
    if count[word] > n:
        if word not in downloaded:
            downloaded[word] = 0
        downloaded[word] += 1

        if downloaded[word] > n:
            continue

        os.makedirs(
            os.path.join("datasets", "bsl_processed_dataset", word), exist_ok=True
        )
        shutil.copy(
            os.path.join("datasets/bsl_dataset", video),
            os.path.join(
                "datasets",
                "bsl_processed_dataset",
                word,
                word + "-" + str(downloaded[word] - 1) + ".mp4",
            ),
        )
