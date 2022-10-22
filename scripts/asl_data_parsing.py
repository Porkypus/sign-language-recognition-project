import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
from constants import DATASET_PATH


def get_videos_ids(json_list):
    """
    function to check if the video id is available in the dataset
    and return the viedos ids of the current instance

    input: instance json list
    output: list of videos_ids

    """
    videos_list = []
    for ins in json_list:
        video_id = ins["video_id"]
        if os.path.exists(f"{DATASET_PATH}videos/{video_id}.mp4"):
            videos_list.append(video_id)
    return videos_list


def get_json_features(json_list):
    """
    function to check if the video id is available in the dataset
    and return the viedos ids and url or any other featrue of the current instance

    input: instance json list
    output: list of videos_ids

    """
    videos_ids = []
    videos_urls = []
    for ins in json_list:
        video_id = ins["video_id"]
        video_url = ins["url"]
        if os.path.exists(f"{DATASET_PATH}videos/{video_id}.mp4"):
            videos_ids.append(video_id)
            videos_urls.append(video_url)
    return videos_ids, videos_urls
